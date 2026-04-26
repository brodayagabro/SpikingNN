import numpy as np
import pandas as pd
import os
from SpikingNN.Izh_net import OneDOFLimb 
from SpikingNN.Izh_net import SimpleAdaptedMuscle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from uuid import uuid4
from scipy.signal import find_peaks, square
import itertools

class lsim:
    def __init__(self, system):
        self.sys = system;

    def simulate(self, y0, T, F_flex, F_ext):        
        dt = T[1]-T[0]
        N = len(T)
        self.sys.set_init_conditions(w0 = y0[0], q0 = y0[1])
        W = np.zeros(N)
        Q = np.zeros(N)
        for i in range(N):
            W[i], Q[i] = self.sys.step(
                dt=dt, 
                F_flex = F_flex[i],
                F_ext = F_ext[i]
            )
        return W, Q

def gen_force(T, duty, w, A = 15, df=0):
    dt = T[1]-T[0]
    w*=0.001
    V = (square(w * T + df, duty=duty)+1.0)
    muscle = SimpleAdaptedMuscle(N=1, w=0.5)
    muscle.set_init_conditions()
    F = np.zeros_like(T)
    for i, t in enumerate(T):
        F[i] = muscle.F
        muscle.step(dt=dt, u=V[i])
    return A*F, V

# Creating object
Limb = OneDOFLimb(m=0.3, ls=0.3, a1=0.2, a2=0.07, q0=np.pi/2)
own_freq = np.sqrt(3*9.81/(2*Limb.ls))
Omega = np.sqrt(own_freq**2 - Limb.b**2/Limb.J**2)
print(f"Собственная частота колебаний: {Omega} рад/с")
Tmax = 100000
print(f'Полное время одной симуляции {Tmax/1000} сек.')

def proc_func(A, df, w):
    # Генерируем уникальный ID для этой симуляции
    sim_id = str(uuid4())
    
    # Создаем словарь с параметрами симуляции
    params = {
        'id': sim_id,
        'A': A,
        'df': df,
        'w': w,
        #'duty': duty,
        #'w0': y0[0],
        #'q0': y0[1]
    }
    
    # Выполняем симуляцию
    LSIM = lsim(
        OneDOFLimb(
            m=0.3,
            ls=0.3,
            a1=0.2,
            a2=0.07,
            q0=np.pi/2, 
            w0=0
        )
    )
    time_scale = 5
    #w = 0.001  # 1 Hz
    duty =0.05
    y0 = [0, np.pi/2]
    T = np.linspace(0, Tmax, Tmax*time_scale)
    F_flex, _ = gen_force(T, duty, w, A = A, df = -df/2)
    F_ext, _ = gen_force(T, duty, w, A = A, df = df/2)
    W, Q = LSIM.simulate(y0, T, F_flex, F_ext)
    
    # Сохраняем результаты в файл
    results_filename = f"sim_{sim_id}.npz"
    results_path = os.path.join(output_dir, results_filename)
    np.savez(results_path, W=W, Q=Q, T=T, F_flex=F_flex, F_ext=F_ext)

    # Осцилограмма
    #img_filename = f"img_{sim_id}.png"
    #img_path = os.path.join(img_dir, img_filename)
    
    #plt.figure()
    #plt.plot(T, Q)
    #plt.title(f'A_{A:.2f}_df_{df:.2f}_w_{w:.2f}')
    #plt.xlabel('t, ms')
    #plt.ylabel('Q, rad')
    #plt.savefig(img_path, dpi=300)
    
    # Добавляем информацию о файле с результатами в параметры
    params['results_file'] = results_filename
    #params['img_file'] = img_filename
    return params


results_db="simulations_db.csv"

# Настройки
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

#img_dir = "img"
#os.makedirs(img_dir, exist_ok=True)

# Диапазоны параметров
A_range = np.linspace(0.1, 0.3, 5)          # 10 точек от 0.01 до 10
df_range = np.linspace(0.1, 2*np.pi-0.1, 7)    # 10 точек от -π до π
w_range = np.linspace(Omega*0.1, Omega*1.5, 7)      # 10 точек от 0.0005 до 0.01
#duty_range = np.linspace(0.05, 0.85, 5)
# Фиксированные параметры
fixed_params = {
    'duty': 0.1,
    'y0': np.array([0, np.pi/2])  # Начальные условия
}

def run_simulation(A, df, w, fixed_params):
    """Запускает одну симуляцию и обрабатывает возможные ошибки"""
    try:
        # Собираем все параметры вместе
        params = {
            'A': float(A),
            'df': float(df),
            'w': float(w),
            #'duty': float(duty)
            **fixed_params
        }
        
        # Запускаем симуляцию
        result = proc_func(**params)
        return result
        
    except Exception as e:
        print(f"\nОшибка при A={A:.3f}, df={df:.3f}, w={w:.5f}: {str(e)}")
        return None

def main():
    # Создаем все комбинации параметров
    param_combinations = list(itertools.product(A_range, df_range, w_range))
    total_simulations = len(param_combinations)
    
    print(f"Всего будет выполнено {total_simulations} симуляций")
    print(f"Диапазоны параметров:")
    print(f"A: от {A_range[0]} до {A_range[-1]}")
    print(f"df: от {df_range[0]} до {df_range[-1]}")
    print(f"w: от {w_range[0]} до {w_range[-1]}")
    
    # Запускаем параллельные вычисления
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(proc_func)(A, df, w)
        for A, df, w in tqdm(param_combinations, desc="Выполнение симуляций")
    )
    
    # Фильтруем успешные результаты
    successful_results = [r for r in results if r is not None]
    failed_count = total_simulations - len(successful_results)
    
    print("\nРезультаты:")
    print(f"Успешно: {len(successful_results)}")
    print(f"Не удалось: {failed_count}")
    
    return successful_results

if __name__ == "__main__":
    simulation_results = main()
    
    # Дополнительно: сохраняем метаданные всех успешных симуляций
    if simulation_results:
        import pandas as pd
        df_results = pd.DataFrame(simulation_results)
        df_results.to_csv(os.path.join(output_dir, "simulations_summary.csv"), index=False)
        print(f"\nМетаданные сохранены в {output_dir}/simulations_summary.csv")
        print(df_results.info())
