"""Shared real-time notebook views; settings are per instance and physics is delegated.

Stop/resume preserves state. Snapshots are explicit user actions. Importing this
module does not start a simulation or load a trained result.
"""
import asyncio
from collections import deque
from datetime import datetime
from html import escape
from io import BytesIO
import math
import os
from pathlib import Path
import sys
import time

import ipywidgets as widgets
from IPython.display import display
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import numpy as np

from types import SimpleNamespace
from .D_trajectory.controller import AngleController, load_solution
from .D_trajectory.trajectory import step_count, stream

DEFAULT_SETTINGS = {'ANNOTATION_SIZE': 11,
 'COMMAND_INTERVAL_S': 0.01,
 'EXTENSOR_COLOR': '#1764c0',
 'FIGSIZE': (14, 9),
 'FLEXOR_COLOR': '#d62728',
 'FONT_FAMILY': 'Times New Roman',
 'FONT_SIZE': 14,
 'FORCE_SCALE_M_PER_N': 0.004,
 'FPS': 15,
 'FRAME_DPI': 95,
 'GRAVITY_MIN_ARROW_M': 0.06,
 'INITIAL_ANGLE_DEG': 0.0,
 'INITIAL_SPEED_RAD_S': 0.0,
 'INITIAL_TARGET_DEG': 0.0,
 'LABEL_SIZE': 14,
 'LEGEND_SIZE': 13,
 'LIMB_COLOR': '#000000',
 'MAX_ARROW_M': 0.16,
 'MAX_BLOCKS_PER_FRAME': 10,
 'MAX_DISPLAY_POINTS': 2400,
 'NEURON_VIEW': 'membrane',
 'SNAPSHOT_DPI': 220,
 'TARGET_COLOR': '#d62728',
 'TARGET_STEP_DEG': 0.5,
 'TICK_SIZE': 13,
 'TIME_SCALE': 1.0,
 'TITLE_SIZE': 15,
 'WIDGET_FONT_SIZE': 15,
 'WINDOW_S': 0.5}

class RecordingAngleController(AngleController):
    """Record each physical step using the same compiled integrator as AngleController.

    Only recording stride changes. Currents, integration order and state updates
    remain identical. ``last_trace`` includes the block's initial observation.
    """

    def reset(self, initial_angle_deg=0., initial_velocity_rad_s=0., target_angle_deg=0.):
        """Reset model state and discard the previous block's recording."""
        snapshot = super().reset(initial_angle_deg, initial_velocity_rad_s, target_angle_deg)
        self.last_trace = np.empty((0, 26))
        return snapshot

    def advance(self, target_angle_deg=None, duration_s=.01):
        """Advance one held command, recording all 0.1 ms samples and spike events."""
        steps = step_count(duration_s)
        if self._failure:
            raise RuntimeError('Simulation stopped; call reset before advancing again')
        if target_angle_deg is not None:
            self.target_angle_deg = target_angle_deg
        reference = np.zeros((steps+1, 2))
        reference[:, 0] = np.deg2rad(self._target)
        # Reuse the controller's complete hidden state, rather than replaying episodes.
        result = stream(self._physical, self._coefficients, self._mechanics,
                        self._state, self._current, reference, self._tau, 0, 1, self._steps)
        self.last_trace = result[1]
        self._steps += result[4]
        self._failure = result[5]
        return self.snapshot()


class LivePendulumActivitySimulation:
    """Display a continuously evolving afferent controller with an online angle input.

    The asyncio task yields between frames, allowing Jupyter widget callbacks
    to run. Stop/resume preserves every neural, muscle and mechanical state.
    Muscle arrows show tensile forces along tendons toward the fixed anchors.
    Their generalized moments follow M_i = -F_i*dL_i/dq in the corrected model.
    """

    def __init__(self, genome_path, snapshot_dir, *, settings=None, activity=True):
        """Load the genome, warm up Numba, build controls and draw the reset state."""
        values = dict(DEFAULT_SETTINGS)
        if settings:
            unknown = set(settings)-set(values)
            if unknown:
                raise ValueError(f'Unknown UI settings: {sorted(unknown)}')
            values.update(settings)
        self.settings = SimpleNamespace(**values)
        self.activity = activity
        self.font = self.settings.FONT_FAMILY
        for directory in (Path.home()/'.local/share/fonts', Path.home()/'.wine/drive_c/windows/Fonts'):
            for path in directory.rglob('times*.ttf'):
                font_manager.fontManager.addfont(str(path))
        if not any(f.name == self.font for f in font_manager.fontManager.ttflist):
            self.font = 'DejaVu Serif'
        ticks = self.settings.COMMAND_INTERVAL_S / .0001
        if (not np.isfinite([self.settings.COMMAND_INTERVAL_S, self.settings.WINDOW_S, self.settings.FPS, self.settings.TIME_SCALE,
                             self.settings.FORCE_SCALE_M_PER_N, self.settings.MAX_ARROW_M, self.settings.GRAVITY_MIN_ARROW_M]).all()
                or self.settings.COMMAND_INTERVAL_S < .0001 or not np.isclose(ticks, round(ticks), rtol=0, atol=1e-7)
                or min(self.settings.WINDOW_S, self.settings.FPS, self.settings.TIME_SCALE, self.settings.FORCE_SCALE_M_PER_N, self.settings.MAX_ARROW_M, self.settings.GRAVITY_MIN_ARROW_M) <= 0
                or self.settings.GRAVITY_MIN_ARROW_M > self.settings.MAX_ARROW_M
                or not isinstance(self.settings.MAX_BLOCKS_PER_FRAME, int) or self.settings.MAX_BLOCKS_PER_FRAME < 1):
            raise ValueError('Invalid timing or arrow settings; commands must use the 0.1 ms grid')
        if self.settings.NEURON_VIEW not in ('membrane', 'spikes') or not isinstance(self.settings.MAX_DISPLAY_POINTS, int) or self.settings.MAX_DISPLAY_POINTS < 4:
            raise ValueError('Invalid neural view or display point budget')
        self.genome_path = genome_path if isinstance(genome_path, dict) else Path(genome_path)
        self.snapshot_dir = Path(snapshot_dir)
        cfg, physical, _, _, _ = load_solution(self.genome_path)
        self.mass, self.length, _, self.gravity = cfg.mechanics
        self.a1, self.a2 = physical[46:48]
        # Both muscles may receive each neuron; color denotes its strongest projection.
        dominant_muscles = np.argmax(physical[48:52].reshape(2, 2), axis=0)
        self.neuron_colors = tuple((self.settings.FLEXOR_COLOR, self.settings.EXTENSOR_COLOR)[m] for m in dominant_muscles)
        self.neuron_labels = tuple(f'N{i} ({("Flexor", "Extensor")[m]})'
                                   for i, m in enumerate(dominant_muscles))
        self.controller = RecordingAngleController(self.genome_path, initial_angle_deg=self.settings.INITIAL_ANGLE_DEG,
                                           initial_velocity_rad_s=self.settings.INITIAL_SPEED_RAD_S,
                                           target_angle_deg=self.settings.INITIAL_TARGET_DEG)
        self.controller.advance(self.settings.INITIAL_TARGET_DEG, self.settings.COMMAND_INTERVAL_S)  # Compile before timing starts.
        self.controller.reset(self.settings.INITIAL_ANGLE_DEG, self.settings.INITIAL_SPEED_RAD_S, self.settings.INITIAL_TARGET_DEG)
        self.history = deque(maxlen=math.ceil(self.settings.WINDOW_S / self.settings.COMMAND_INTERVAL_S) + 3)
        self.activity_history = deque()  # Chunks: time, V0, V1, afferent sums F/E, forces F/E (N).
        self.spike_times = [deque(), deque()]  # Exact event times for the two model neurons.
        self._record_activity()
        self._task, self.running, self.closed = None, False, False
        self.last_error, self.last_snapshot_path = None, None
        self._record()
        self.target_slider = widgets.FloatSlider(
            value=self.settings.INITIAL_TARGET_DEG, min=-45., max=45., step=self.settings.TARGET_STEP_DEG,
            description='Цель, °:', continuous_update=True, readout_format='.1f',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='470px'))
        self.start_stop = widgets.Button(description='Старт', icon='play', button_style='success')
        self.reset_button = widgets.Button(description='Сброс', icon='undo')
        self.snapshot_button = widgets.Button(description='Снимок PNG', icon='camera')
        self.neuron_view = widgets.ToggleButtons(
            options=[('Потенциалы, мВ', 'membrane'), ('Растр спайков', 'spikes')],
            value=self.settings.NEURON_VIEW, description='Нейроны:', style={'description_width': 'initial'})
        self.readout = widgets.HTML()
        self.message = widgets.HTML()
        self.image = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1450px'))
        controls = widgets.HBox([self.start_stop, self.reset_button, self.snapshot_button, self.target_slider],
                                 layout=widgets.Layout(flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, self.neuron_view, self.readout, self.image, self.message])
        self.neuron_view.observe(self._neuron_view_changed, names='value')
        self.target_slider.observe(self._target_changed, names='value')
        self.start_stop.on_click(self._toggle)
        self.reset_button.on_click(self._reset_clicked)
        self.snapshot_button.on_click(self._snapshot_clicked)
        self._make_figure()
        self._draw()
        self._notice('Загружено явно выбранное решение. Старт/стоп сохраняет состояние системы.')

    def _make_figure(self):
        """Create persistent artists once; later frames only update their data."""
        style = {'font.family': self.font, 'font.size': self.settings.FONT_SIZE, 'axes.titlesize': self.settings.TITLE_SIZE,
                 'axes.labelsize': self.settings.LABEL_SIZE, 'xtick.labelsize': self.settings.TICK_SIZE,
                 'ytick.labelsize': self.settings.TICK_SIZE, 'legend.fontsize': self.settings.LEGEND_SIZE, 'mathtext.fontset': 'stix'}
        with mpl.rc_context(style):
            self.figure = Figure(figsize=self.settings.FIGSIZE, dpi=self.settings.FRAME_DPI, layout='constrained')
            self.canvas = FigureCanvasAgg(self.figure)
            # Fit all four time plots within the existing figure height.
            grid = self.figure.add_gridspec(4, 2, width_ratios=[1., 1.6])
            self.scene = self.figure.add_subplot(grid[:, 0])
            self.chart = self.figure.add_subplot(grid[0, 1])
            self.neuron_chart = self.figure.add_subplot(grid[1, 1], sharex=self.chart)
            self.afferent_chart = self.figure.add_subplot(grid[2, 1], sharex=self.chart)
            self.muscle_chart = self.figure.add_subplot(grid[3, 1], sharex=self.chart)
            ax = self.scene
            extent = max(self.length, self.a1 + .06, self.a2 + self.settings.MAX_ARROW_M) * 1.12
            ax.set(xlim=(-extent, extent), ylim=(-self.length*.65, self.length*1.28),
                   aspect='equal', title='')
            ax.axis('off')
            ax.plot([-extent*.9, extent*.9], [0, 0], color='#a7a7a7', lw=1, zorder=0)
            ax.plot([0, 0], [0, self.length*1.06], ':', color='#b6b6b6', lw=1)
            self.target_line, = ax.plot([], [], '-', color=self.settings.TARGET_COLOR, lw=2, zorder=6, label='Target')
            self.flexor_line, = ax.plot([], [], color=self.settings.FLEXOR_COLOR, lw=2.5, label='Flexor')
            self.extensor_line, = ax.plot([], [], color=self.settings.EXTENSOR_COLOR, lw=2.5, label='Extensor')
            self.limb_line, = ax.plot([], [], '-o', color=self.settings.LIMB_COLOR, lw=4, markersize=5, label='Limb')
            ax.plot([0], [0], 'o', color='black', ms=8, zorder=10)
            ax.plot([self.a1], [0], 's', color=self.settings.FLEXOR_COLOR, ms=5)
            ax.plot([-self.a1], [0], 's', color=self.settings.EXTENSOR_COLOR, ms=5)
            self.force_arrows = []
            for color in (self.settings.FLEXOR_COLOR, self.settings.EXTENSOR_COLOR, '#9b6818'):
                arrow = FancyArrowPatch((0, 0), (.01, 0), arrowstyle='-|>',
                                         mutation_scale=15, lw=2, color=color, zorder=8)
                ax.add_patch(arrow)
                self.force_arrows.append(arrow)
            self.gravity_label = ax.text(0, 0, 'mg', color='#9b6818', fontsize=self.settings.ANNOTATION_SIZE)
            self.force_text = ax.text(.01, .17, '', transform=ax.transAxes,
                                      fontsize=self.settings.ANNOTATION_SIZE, va='bottom')
            self.angle_text = ax.text(.02, .96, '', transform=ax.transAxes,
                                      fontsize=self.settings.ANNOTATION_SIZE, va='top')
            ax.legend(loc='lower left', bbox_to_anchor=(.0, .0), ncol=2, framealpha=.85)
            self.target_trace, = self.chart.plot([], [], '--', color=self.settings.TARGET_COLOR, lw=2,
                                                 drawstyle='steps-post', label='Target angle')
            self.angle_trace, = self.chart.plot([], [], color=self.settings.LIMB_COLOR, lw=1.5, label='Limb`s angle')
            self.chart.set(title='', xlabel='',
                            ylabel='Angle, °', xlim=(0, self.settings.WINDOW_S), ylim=(-55, 55))
            self.voltage_traces, self.spike_traces, self.afferent_traces = [], [], []
            self.muscle_traces = []
            for i, color in enumerate(self.neuron_colors):
                line, = self.neuron_chart.plot([], [], color=color, lw=.8, label=self.neuron_labels[i])
                raster, = self.neuron_chart.plot([], [], '|', color=color, ms=10, ls='none')
                self.voltage_traces.append(line)
                self.spike_traces.append(raster)
            for label, color in (('Flexor', self.settings.FLEXOR_COLOR), ('Extensor', self.settings.EXTENSOR_COLOR)):
                line, = self.afferent_chart.plot([], [], color=color, lw=1.2, label=label)
                self.afferent_traces.append(line)
                force, = self.muscle_chart.plot([], [], color=color, lw=1.2, label=label)
                self.muscle_traces.append(force)
            self.neuron_chart.legend(handles=self.voltage_traces, loc='upper right', ncol=2)
            self.afferent_chart.set(ylabel='Ia + II + Ib\n(model units)', ylim=(0, 1))
            self.afferent_chart.legend(loc='upper right', ncol=2)
            self.muscle_chart.set(xlabel='t, s', ylabel='Force, N', ylim=(0, 1))
            self.muscle_chart.legend(loc='upper right', ncol=2)
            self.chart.tick_params(labelbottom=False)
            self.neuron_chart.tick_params(labelbottom=False)
            self.afferent_chart.tick_params(labelbottom=False)
            self.neuron_chart.grid(alpha=.25)
            self.afferent_chart.grid(alpha=.25)
            self.muscle_chart.grid(alpha=.25)
            self._configure_neuron_axis()
            # Bold panel labels stay attached to their respective axes.
            for panel, label in ((self.scene, 'A'), (self.chart, 'B'),
                                 (self.neuron_chart, 'C'), (self.afferent_chart, 'D'),
                                 (self.muscle_chart, 'E')):
                panel.set_title(label, loc='left', fontweight='bold', fontsize=self.settings.TITLE_SIZE)
            self.chart.grid(alpha=.25)
            self.chart.legend(loc='upper right')
            if not self.activity:
                for panel in (self.neuron_chart, self.afferent_chart, self.muscle_chart):
                    panel.remove()
                self.chart.set_subplotspec(grid[:, 1])
                self.chart.tick_params(labelbottom=True)
                self.chart.set_xlabel('t, s')
                self.neuron_view.layout.display = 'none'
            self.figure.supxlabel('Muscle arrows: tendon tension toward anchors. '
                                   'mg: gravity (enlarged for visibility). Model: v3-tensile.',
                                   fontsize=max(8, self.settings.ANNOTATION_SIZE-1), fontfamily=self.font)

    def _configure_neuron_axis(self):
        """Switch between membrane potentials and exact spike events without resetting state."""
        membrane = self.neuron_view.value == 'membrane'
        for voltage, raster in zip(self.voltage_traces, self.spike_traces):
            voltage.set_visible(membrane)
            raster.set_visible(not membrane)
        if membrane:
            self.neuron_chart.set_ylabel('V, mV')
            self.neuron_chart.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        else:
            self.neuron_chart.set(ylabel='Neuron', ylim=(-.6, 1.6))
            self.neuron_chart.set_yticks([0, 1], labels=['N0', 'N1'])

    def _neuron_view_changed(self, _change):
        """Apply the display mode immediately; keep the same physical trajectory."""
        self._configure_neuron_axis()
        if not self.running:
            self._draw()

    def _record_activity(self, trace=None):
        """Retain voltages, afferent sums, forces and spikes for the configured window."""
        if trace is None:
            state = self.controller.snapshot()
            sums = state['afferents'].reshape(2, 3).sum(axis=1)
            rows = np.array([[state['time_s'], *state['membrane_mV'], *sums, *state['force_N']]])
        else:
            # Skip the block's initial row to avoid repeating samples and spikes.
            post = trace[1:]
            if not len(post):
                return
            rows = np.column_stack((post[:, 0], post[:, 5:7],
                                    post[:, 11:14].sum(axis=1), post[:, 14:17].sum(axis=1),
                                    post[:, 3:5]))  # Actual flexor/extensor tension in newtons.
            for i in range(2):
                self.spike_times[i].extend(post[post[:, 19+i] > 0, 0])
        self.activity_history.append(rows)
        cutoff = rows[-1, 0]-self.settings.WINDOW_S
        while len(self.activity_history) > 1 and self.activity_history[0][-1, 0] < cutoff:
            self.activity_history.popleft()
        # Preserve at most one sample preceding the left edge for continuous curves.
        first = self.activity_history[0]
        skip = max(0, int(np.searchsorted(first[:, 0], cutoff))-1)
        if skip:
            self.activity_history[0] = first[skip:].copy()
        for events in self.spike_times:
            while events and events[0] < cutoff:
                events.popleft()

    def _display_envelope(self, times, values):
        """Reduce drawing cost with time-ordered extrema; never decimate the recorded data.

        Each bin contributes its minimum and maximum. Peaks therefore remain
        visible even when they last one integration step. The spike raster uses
        all event times directly and does not use this display reduction.
        """
        if len(values) <= self.settings.MAX_DISPLAY_POINTS:
            return times, values
        width = math.ceil(len(values) / max(1, (self.settings.MAX_DISPLAY_POINTS-4)//2))
        groups = len(values)//width
        blocks = values[:groups*width].reshape(groups, width)
        bases = np.arange(groups)*width
        indices = np.column_stack((bases+blocks.argmin(axis=1), bases+blocks.argmax(axis=1))).ravel()
        if groups*width < len(values):
            tail = values[groups*width:]
            indices = np.r_[indices, groups*width+tail.argmin(), groups*width+tail.argmax()]
        indices = np.unique(np.r_[0, indices, len(values)-1])
        return times[indices], values[indices]

    def geometry(self, state=None):
        """Return diagram geometry and force vectors consistent with model moments.

        Positive theta is clockwise from the upward vertical: r=(sin(theta), cos(theta)).
        Tendon lengths match L(q) at q=theta-pi/2. Positive tension pulls the
        attachment toward its anchor; moment is force dotted with dr/dtheta.
        Gravity acts downward at the midpoint of the uniform rod.
        """
        state = self.controller.snapshot() if state is None else state
        theta = np.deg2rad(state['angle_deg'])
        direction = np.array([np.sin(theta), np.cos(theta)])
        tangent = np.array([np.cos(theta), -np.sin(theta)])
        attachment = self.a2*direction
        anchors = np.array([[self.a1, 0.], [-self.a1, 0.]])
        lengths = np.linalg.norm(anchors-attachment, axis=1)
        # A muscle applies tension toward its fixed anchor, never a pushing force.
        muscle_vectors = state['force_N'][:, None]*(anchors-attachment)/lengths[:, None]
        moments = muscle_vectors @ (self.a2*tangent)
        gravity_vector = np.array([0., -self.mass*self.gravity])
        return dict(direction=direction, tangent=tangent, attachment=attachment, anchors=anchors,
                    lengths=lengths, moments=moments, muscle_vectors=muscle_vectors,
                    gravity_vector=gravity_vector, center=.5*self.length*direction)

    def _record(self):
        """Retain the rolling window and one preceding sample for a continuous boundary."""
        s = self.controller.snapshot()
        row = (s['time_s'], s['target_angle_deg'], s['angle_deg'])
        if self.history and abs(self.history[-1][0]-row[0]) < 1e-12:
            self.history[-1] = row
        else:
            self.history.append(row)
        while len(self.history) > 2 and self.history[1][0] < row[0]-self.settings.WINDOW_S:
            self.history.popleft()

    def _advance_block(self):
        """Advance one command interval; no neuron, muscle or limb state is reset."""
        s = self.controller.advance(float(self.target_slider.value), self.settings.COMMAND_INTERVAL_S)
        if s['failure'] == 'nonfinite':
            raise FloatingPointError('Нечисловое состояние; требуется сброс.')
        self._record()
        self._record_activity(self.controller.last_trace)
        return s

    def _draw_arrow(self, artist, origin, force, min_length=0.):
        """Apply visual length limits while preserving force direction and physical values."""
        delta = self.settings.FORCE_SCALE_M_PER_N*np.asarray(force)
        length = np.linalg.norm(delta)
        artist.set_visible(length > 1e-10)
        if length > 1e-10:
            delta *= np.clip(length, min_length, self.settings.MAX_ARROW_M)/length
        artist.set_positions(origin, origin+delta)
        return delta

    def _draw(self):
        """Render the real current state and rolling traces into the notebook image widget."""
        s = self.controller.snapshot()
        if not np.isfinite([s['angle_deg'], s['velocity_rad_s'], *s['force_N']]).all():
            return  # Keep the last valid frame if the numerical model fails.
        g = self.geometry(s)
        target = np.deg2rad(self.target_slider.value)
        # Extend the target slightly so it stays visible when the limb reaches it.
        target_tip = 1.12*self.length*np.array([np.sin(target), np.cos(target)])
        limb_tip = self.length*g['direction']
        self.target_line.set_data([0, target_tip[0]], [0, target_tip[1]])
        self.limb_line.set_data([0, limb_tip[0]], [0, limb_tip[1]])
        for line, anchor in zip((self.flexor_line, self.extensor_line), g['anchors']):
            line.set_data([anchor[0], g['attachment'][0]], [anchor[1], g['attachment'][1]])
        for arrow, force in zip(self.force_arrows[:2], g['muscle_vectors']):
            self._draw_arrow(arrow, g['attachment'], force)
        gravity_delta = self._draw_arrow(
            self.force_arrows[2], g['center'], g['gravity_vector'], min_length=self.settings.GRAVITY_MIN_ARROW_M)
        self.gravity_label.set_position(g['center']+.5*gravity_delta+np.array([.012, 0.]))
        ff, fe = s['force_N']
        mf, me = g['moments']
        self.force_text.set_text(f'Силы мышц: Fсг = {ff:.1f} Н; Fразг = {fe:.1f} Н\n'
                                 f'Моменты: Mсг = {mf:+.3f} Н·м; Mразг = {me:+.3f} Н·м')
        self.angle_text.set_text(f't = {s["time_s"]:.2f} с\nθ* = {self.target_slider.value:+.1f}°\n'
                                 f'θ = {s["angle_deg"]:+.1f}°')
        data = np.asarray(self.history)
        self.target_trace.set_data(data[:, 0], data[:, 1])
        self.angle_trace.set_data(data[:, 0], data[:, 2])
        right = max(self.settings.WINDOW_S, s['time_s'])
        self.chart.set_xlim(right-self.settings.WINDOW_S, right)
        limit = max(55., float(np.max(abs(data[:, 1:]))+5.))
        self.chart.set_ylim(-limit, limit)
        activity = np.concatenate(tuple(self.activity_history))
        for i, (voltage, raster, afferent, muscle) in enumerate(zip(
                self.voltage_traces, self.spike_traces, self.afferent_traces, self.muscle_traces)):
            voltage.set_data(*self._display_envelope(activity[:, 0], activity[:, 1+i]))
            events = np.asarray(self.spike_times[i])
            raster.set_data(events, np.full(len(events), i))
            afferent.set_data(*self._display_envelope(activity[:, 0], activity[:, 3+i]))
            muscle.set_data(*self._display_envelope(activity[:, 0], activity[:, 5+i]))
        if self.neuron_view.value == 'membrane':
            self.neuron_chart.set_ylim(min(-90., float(activity[:, 1:3].min())-5.),
                                       max(35., float(activity[:, 1:3].max())+5.))
        self.afferent_chart.set_ylim(0., max(1., 1.1*float(activity[:, 3:5].max())))
        self.muscle_chart.set_ylim(0., max(1., 1.1*float(activity[:, 5:7].max())))
        state_label = 'Работает' if self.running else 'Пауза'
        self.readout.value = (f'<div style="font-family:{escape(self.font)};font-size:{self.settings.WIDGET_FONT_SIZE}px">'
                              f'<b>{state_label}</b> · t = {s["time_s"]:.2f} с · '
                              f'цель = {self.target_slider.value:+.1f}° · угол = {s["angle_deg"]:+.1f}° · '
                              f'скорость = {s["velocity_rad_s"]:+.2f} рад/с</div>')
        # A PNG widget avoids backend switching and works in JupyterLab and Notebook.
        buffer = BytesIO()
        self.canvas.print_png(buffer)
        self.image.value = buffer.getvalue()

    def _notice(self, text, error=False):
        """Show a compact status or saved-image path, with HTML escaping."""
        color = '#a32020' if error else '#555555'
        self.message.value = (f'<span style="color:{color};font-size:{self.settings.WIDGET_FONT_SIZE}px">'
                               f'{escape(str(text))}</span>')

    def _target_changed(self, change):
        """Update the command immediately while preserving the actual limb state."""
        self.controller.target_angle_deg = float(change['new'])
        self._record()
        if not self.running:
            self._draw()

    def _toggle(self, _button):
        """Handle the combined start/stop button."""
        self.stop() if self.running else self.start()

    def start(self):
        """Start or resume without resetting state or counting time spent paused."""
        if self.closed or self.running:
            return
        if self.controller.snapshot()['failure'] != 'none':
            self._notice('Расчёт остановлен после падения. Нажмите «Сброс».', error=True)
            return
        self.running = True
        self._wall_origin = time.perf_counter()
        self._sim_origin = self.controller.snapshot()['time_s']
        self.start_stop.description, self.start_stop.icon, self.start_stop.button_style = 'Стоп', 'pause', 'warning'
        self._notice('Меняйте целевой угол ползунком. Снимок сохраняет все пять панелей и ставит расчёт на паузу.')
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self):
        """Track wall time with bounded catch-up and yield regularly to widget callbacks."""
        task = asyncio.current_task()
        try:
            while self.running and self._task is task:
                started = time.perf_counter()
                desired_elapsed = (started-self._wall_origin)*self.settings.TIME_SCALE
                actual_elapsed = self.controller.snapshot()['time_s']-self._sim_origin
                blocks = min(self.settings.MAX_BLOCKS_PER_FRAME,
                             max(0, int((desired_elapsed-actual_elapsed+1e-10)/self.settings.COMMAND_INTERVAL_S)))
                for _ in range(blocks):
                    s = self._advance_block()
                    if s['failure'] != 'none':
                        self.running = False
                        self._notice('Конечность вышла за ±90°. Расчёт остановлен; нажмите «Сброс».', error=True)
                        break
                self._draw()
                # Every 0.1 ms physical step is retained. Slow rendering can cause lag,
                # but elapsed simulation time is never fabricated or skipped forward.
                await asyncio.sleep(max(.001, 1/self.settings.FPS-(time.perf_counter()-started)))
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self.last_error = error
            self._notice(f'Расчёт остановлен: {error}', error=True)
        finally:
            if self._task is task:
                self._task, self.running = None, False
                self.start_stop.description, self.start_stop.icon, self.start_stop.button_style = 'Старт', 'play', 'success'
                self._draw()

    def stop(self):
        """Freeze the model and chart; the next start resumes from this exact state."""
        self.running = False
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        self.start_stop.description, self.start_stop.icon, self.start_stop.button_style = 'Старт', 'play', 'success'
        if not self.closed:
            self._draw()

    def _reset_clicked(self, _button):
        """Reset the physical state and history explicitly, retaining the current slider target."""
        self.stop()
        self.controller.reset(self.settings.INITIAL_ANGLE_DEG, self.settings.INITIAL_SPEED_RAD_S, float(self.target_slider.value))
        self.history.clear()
        self.activity_history.clear()
        for events in self.spike_times:
            events.clear()
        self._record_activity()
        self.last_error = None
        self._record()
        self._draw()
        self._notice('Начальное состояние восстановлено. Нажмите «Старт».')

    def save_snapshot(self):
        """Pause and save a high-resolution PNG of the schematic and all four rolling plots."""
        self.stop()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = self.snapshot_dir / f'pendulum_activity_{stamp}_t{self.controller.snapshot()["time_s"]:.2f}s.png'
        self.figure.savefig(path, dpi=self.settings.SNAPSHOT_DPI, bbox_inches='tight')
        self.last_snapshot_path = path
        self._notice(f'Пауза. Снимок сохранён: {path}')
        return path

    def _snapshot_clicked(self, _button):
        """Report snapshot errors in the widget instead of losing the button callback."""
        try:
            self.save_snapshot()
        except Exception as error:
            self._notice(f'Не удалось сохранить снимок: {error}', error=True)

    def close(self):
        """Cancel the background task and release this cell's old widget instance."""
        if self.closed:
            return
        self.stop()
        self.closed = True
        self.neuron_view.unobserve(self._neuron_view_changed, names='value')
        self.target_slider.unobserve(self._target_changed, names='value')
        for widget in (self.target_slider, self.start_stop, self.reset_button, self.snapshot_button,
                       self.neuron_view, self.readout, self.message, self.image, self.ui):
            widget.close()
        self.figure.clear()


