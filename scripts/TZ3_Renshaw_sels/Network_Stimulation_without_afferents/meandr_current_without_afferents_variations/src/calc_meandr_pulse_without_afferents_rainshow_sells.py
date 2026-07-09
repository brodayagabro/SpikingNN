import numpy as np
from SpikingNN.core.Izh_net import *
from SpikingNN.Var_Limb import *
from Rybak2002 import *
from tqdm import tqdm
import os
import pandas as pd
from joblib import Parallel, delayed
import logging
from pathlib import Path
from feature_extractor import *
from uuid import uuid4

# ==========================================================
# 1. ИНИЦИАЛИЗАЦИЯ И ЛОГИРОВАНИЕ
# ==========================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_model():
    Flexor = SimpleAdaptedMuscle(w=0.5, N=50)
    Extensor = SimpleAdaptedMuscle(w=0.5, N=50)
    Flexor.tau_1 = 1/21; Flexor.tau_c = 1/39
    Extensor.tau_1 = 1/21; Extensor.tau_c = 1/39
    
    Pendulum = OneDOFLimb(q0=np.pi/2, w0=0., b=0.01, a1=0.4, a2=0.05, m=0.3, l=0.3)
    Limb = Simple_Afferented_Limb(Limb=Pendulum, Flexor=Flexor, Extensor=Extensor)
    
    Qapp = np.zeros((12, 4))
    Qapp[0, 0] = Qapp[1, 1] = Qapp[6, 2] = Qapp[7, 3] = 1
    
    Rybak2002Net = Rybak_2002_network(
        input_size=4, output_size=2, afferent_size=6,
        Qapp=Qapp, exitatory_w=0.5, inhibitory_w=-0.5
    )
    Rybak2002FullSystem = Var_Limb(Network=Rybak2002Net, Limb=Limb)
    Rybak2002FullSystem.set_afferents_by_names('Ia_Flex', 'Ia_IN_Flex', 10)
    Rybak2002FullSystem.set_afferents_by_names('Ia_Ext', 'Ia_IN_Ext', 10)
    return Rybak2002FullSystem

class SourceTargetParser:
    @staticmethod
    def parse(text, delimiter=None):
        if delimiter:
            if delimiter not in text: raise ValueError(f"Delimiter '{delimiter}' not found")
            source, target = text.split(delimiter, 1)
        else:
            if '->' in text: source, target = text.split('->', 1)
            elif '-' in text: source, target = text.split('-', 1)
            else: raise ValueError("Cannot determine delimiter")
        return source.strip(), target.strip()

    @staticmethod
    def parse_arrow(text):
        return SourceTargetParser.parse(text, '->')

# ==========================================================
# 2. ГЕНЕРАТОР ТОКА И ПРИМЕНЕНИЕ ВЕСОВ
# ==========================================================
def make_pulse_generator(params):
    period = [params["pulse_period_ch0"], params["pulse_period_ch1"], 
              params["pulse_period_ch2"], params["pulse_period_ch3"]]
    durations = [params["pulse_duration_ch0"], params["pulse_duration_ch1"], 
                 params["pulse_duration_ch2"], params["pulse_duration_ch3"]]
    amplitudes = [params["amplitude_ch0"], params["amplitude_ch1"], 
                  params["amplitude_ch2"], params["amplitude_ch3"]]
    phases = [params["phase_ch0"], params["phase_ch1"], 
              params["phase_ch2"], params["phase_ch3"]]
    base_currents = [params["base_current_I1"], params["base_current_I2"], 
                     params["base_current_I3"], params["base_current_I4"]]
    noise_percent = params["noise_percent"]

    def Iapp(t):
        I = np.zeros(4)
        for i in range(4):
            t_phase = (t - phases[i] * period[i]) % period[i]
            pulse_value = amplitudes[i] if t_phase < durations[i] else 0.0
            noise = np.random.normal(0, base_currents[i] * noise_percent)
            I[i] = base_currents[i] + pulse_value + noise
        return I
    return Iapp

def apply_all_network_weights(model, params):
    """Применяет ВСЕ веса из конфига, а не только одну пару."""
    conn_cols = [col for col in params.keys() if '->' in col]
    for conn_str in conn_cols:
        try:
            source, target = SourceTargetParser.parse_arrow(conn_str)
            weight = float(params[conn_str])
            model.set_weights_by_names(source, target, weight)
        except Exception as e:
            logger.warning(f"Skipping connection {conn_str}: {e}")

# ==========================================================
# 3. ЗАПУСК СИМУЛЯЦИИ
# ==========================================================
def run_simulation_task(params):
    try:
        logger.info(f"Starting simulation for combination {params['combination_id']}")
        result = dict(params)
        model = initialize_model()

        # НОВОЕ: Применяем все веса из сгенерированного CSV
        apply_all_network_weights(model, params)

        Iapp = make_pulse_generator(params)
        scale = 5
        Tmax = 10000
        T = np.linspace(0, Tmax, scale * Tmax)
        dt = T[1] - T[0]
        N = 12

        V_curr = np.zeros((len(T), N))
        U_curr = np.zeros((len(T), N))
        F_flex_curr = np.zeros(len(T))
        F_ext_curr = np.zeros(len(T))
        Afferents_curr = np.zeros((len(T), 6))
        Q_curr = np.zeros(len(T))
        W_curr = np.zeros(len(T))

        for i, t in enumerate(T): 
            V_curr[i] = model.V
            U_curr[i] = model.U
            F_flex_curr[i] = model.F_flex
            F_ext_curr[i] = model.F_ext
            Afferents_curr[i] = model.Limb.output
            Q_curr[i] = model.q
            W_curr[i] = model.w
            model.step(dt=dt, Iapp=Iapp(t))

        start_index = int(len(T) * 0.5)
        T_trimmed = T[start_index:]
        V_trimmed = V_curr[start_index:]
        U_trimmed = U_curr[start_index:]
        F_flex_trimmed = F_flex_curr[start_index:]
        F_ext_trimmed = F_ext_curr[start_index:]
        Afferents_trimmed = Afferents_curr[start_index:]
        Q_trimmed = Q_curr[start_index:]
        W_trimmed = W_curr[start_index:]

        output_dir = Path("../data")
        output_dir.mkdir(exist_ok=True)
        muscle_file_id = uuid4()
        muscle_csv_filename = f"meandr_pulses_without_afferents_{muscle_file_id}_muscles.csv"
        muscle_csv_path = output_dir / muscle_csv_filename

        pd.DataFrame({
            't': T_trimmed, 'F_flex': F_flex_trimmed, 'F_ext': F_ext_trimmed
        }).to_csv(muscle_csv_path, index=False)

        features = compute_all_features(
            F_flex_trimmed, F_ext_trimmed, V_trimmed, U_trimmed, 
            Afferents_trimmed, Q_trimmed, W_trimmed, T_trimmed
        )
        mode = classify_operation_mode(features)

        result.update(features)
        result['mode'] = mode
        result.update({"muscle_csv_file": str(muscle_csv_filename), "status": "success"})
        logger.info(f"Successfully completed combination {params['combination_id']}")
        return result

    except Exception as e:
        logger.error(f"Error in combination {params['combination_id']}: {str(e)}")
        result = dict(params)
        result.update({"muscle_csv_file": None, "mode": "Error processing", "status": "error"})
        return result

# ==========================================================
# 4. ГЛАВНЫЙ КОНТРОЛЛЕР
# ==========================================================
def main():
    print("="*50)
    print("Simulation Engine for Rybak Network Configs")
    print("="*50)
    
    config_file = "cfg.csv"
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}. Run the generator first.")
        return
        
    df = pd.read_csv(config_file)
    logger.info(f"Loaded {len(df)} configurations.")
    selected_params_list = [row.to_dict() for _, row in df.iterrows()]

    results = Parallel(n_jobs=40, verbose=2)(
        delayed(run_simulation_task)(params) 
        for params in tqdm(selected_params_list, desc="Simulating")
    )

    if results:
        results_df = pd.DataFrame(results)
        out_file = "../experiment_results_meandr_pulse_without_afferents_rainshow.csv"
        results_df.to_csv(out_file, index=False)
        logger.info(f"Results saved to {out_file}")
        
        success_count = results_df[results_df['status'] == 'success'].shape[0]
        error_count = results_df[results_df['status'] == 'error'].shape[0]
        print(f"\n Statistics:")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Interrupted by user")
    except Exception as e:
        print(f"\n Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
