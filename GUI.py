import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io

class NeuronNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuron Network Designer")

        self.neurons = {}
        self.next_neuron_id = 1
        self.connections = []
        self.weight_matrix = None

        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(root, width=200, height=600, bg="#f0f0f0")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        tk.Button(control_frame, text="Добавить нейрон", command=self.add_neuron).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Изменить имя", command=self.rename_neuron).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Создать связь", command=self.start_connection).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Удалить элемент", command=self.delete_element).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Показать матрицу", command=self.show_matrix).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Сохранить матрицу", command=self.save_matrix).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Экспорт в PNG", command=self.export_png).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Экспорт в SVG", command=self.export_svg).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Очистить всё", command=self.clear_all).pack(fill=tk.X, pady=5)

        # Состояния приложения
        self.current_mode = "select"  # Режимы: select, connect
        self.selected_neuron = None
        self.connection_start = None
        self.temp_line = None
        self.dragging = False
        self.drag_item = None

        # Привязка событий
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.handle_drag)
        self.canvas.bind("<ButtonRelease-1>", self.handle_release)
        self.canvas.bind("<Button-3>", self.cancel_operation)

        self.status_var = tk.StringVar()
        self.status_var.set("Готово. Выберите действие")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def handle_click(self, event):
        if self.current_mode == "select":
            self.select_neuron(event)
        elif self.current_mode == "connect":
            self.process_connection(event)

    def handle_drag(self, event):
        if self.dragging and self.drag_item:
            dx = event.x - self.drag_start_pos[0]
            dy = event.y - self.drag_start_pos[1]

            # Перемещаем нейрон
            self.canvas.move(self.drag_item["circle"], dx, dy)
            self.canvas.move(self.drag_item["text"], dx, dy)

            # Обновляем позицию в данных
            self.drag_item["x"] += dx
            self.drag_item["y"] += dy

            # Обновляем связи
            self.update_connections(self.drag_item["id"])

            self.drag_start_pos = (event.x, event.y)

    def handle_release(self, event):
        self.dragging = False
        self.drag_item = None

    def select_neuron(self, event):
        clicked_items = self.canvas.find_overlapping(event.x-20, event.y-20, event.x+20, event.y+20)
        neurons_in_area = [item for item in clicked_items if "neuron" in self.canvas.gettags(item)]

        if neurons_in_area:
            neuron_id = int(self.canvas.gettags(neurons_in_area[0])[1])
            self.selected_neuron = neuron_id
            self.dragging = True
            self.drag_item = self.neurons[neuron_id]
            self.drag_start_pos = (event.x, event.y)
            self.status_var.set(f"Выбран нейрон {neuron_id}. Перетащите для перемещения")
        else:
            self.selected_neuron = None
            self.status_var.set("Ничего не выбрано")

    def start_connection(self):
        self.current_mode = "connect"
        self.status_var.set("Режим создания связи. Выберите начальный нейрон")

    def process_connection(self, event):
        clicked_items = self.canvas.find_overlapping(event.x-20, event.y-20, event.x+20, event.y+20)
        neurons_in_area = [item for item in clicked_items if "neuron" in self.canvas.gettags(item)]

        if not neurons_in_area:
            if self.connection_start:
                # Обновляем временную линию
                x1, y1 = self.neurons[self.connection_start]["x"], self.neurons[self.connection_start]["y"]
                if not self.temp_line:
                    self.temp_line = self.canvas.create_line(x1, y1, event.x, event.y, arrow=tk.LAST, dash=(4, 2))
                else:
                    self.canvas.coords(self.temp_line, x1, y1, event.x, event.y)
            return

        neuron_id = int(self.canvas.gettags(neurons_in_area[0])[1])

        if not self.connection_start:
            # Выбираем начальный нейрон
            self.connection_start = neuron_id
            self.status_var.set(f"Начальный нейрон {neuron_id}. Выберите конечный нейрон")
        else:
            # Завершаем создание связи
            if neuron_id == self.connection_start:
                messagebox.showwarning("Ошибка", "Нельзя соединить нейрон с самим собой")
                self.cancel_operation()
                return

            # Проверяем существующую связь
            for conn in self.connections:
                if conn[0] == self.connection_start and conn[1] == neuron_id:
                    messagebox.showwarning("Ошибка", "Такая связь уже существует")
                    self.cancel_operation()
                    return

            # Запрашиваем вес
            weight = simpledialog.askfloat("Вес связи",
                                         f"Введите вес связи от {self.connection_start} к {neuron_id}:",
                                         initialvalue=1.0)
            if weight is None:
                self.cancel_operation()
                return

            # Создаем связь
            self.create_connection(self.connection_start, neuron_id, weight)
            self.cancel_operation()

    def create_connection(self, from_id, to_id, weight):
        x1, y1 = self.neurons[from_id]["x"], self.neurons[from_id]["y"]
        x2, y2 = self.neurons[to_id]["x"], self.neurons[to_id]["y"]

        # Корректируем позиции для краев кругов
        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            x1 = x1 + (dx/dist)*20
            y1 = y1 + (dy/dist)*20
            x2 = x2 - (dx/dist)*20
            y2 = y2 - (dy/dist)*20

        arrow_shape = "8 10 3" if weight >= 0 else "8 10 3"
        line = self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2, arrowshape=arrow_shape)
        text_x = (x1 + x2) / 2
        text_y = (y1 + y2) / 2
        text = self.canvas.create_text(text_x, text_y, text=f"{weight:.2f}", fill="red")

        self.connections.append((from_id, to_id, weight, line, text))
        self.status_var.set(f"Создана связь {from_id} -> {to_id} с весом {weight:.2f}")

    def update_connections(self, neuron_id):
        for conn in self.connections:
            if conn[0] == neuron_id or conn[1] == neuron_id:
                from_id, to_id, weight, line, text = conn

                # Удаляем старые элементы
                self.canvas.delete(line)
                self.canvas.delete(text)

                # Создаем новые с обновленными координатами
                x1, y1 = self.neurons[from_id]["x"], self.neurons[from_id]["y"]
                x2, y2 = self.neurons[to_id]["x"], self.neurons[to_id]["y"]

                dx = x2 - x1
                dy = y2 - y1
                dist = (dx**2 + dy**2)**0.5
                if dist > 0:
                    x1 = x1 + (dx/dist)*20
                    y1 = y1 + (dy/dist)*20
                    x2 = x2 - (dx/dist)*20
                    y2 = y2 - (dy/dist)*20

                arrow_shape = "8 10 3" if weight >= 0 else "8 10 3"
                new_line = self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2, arrowshape=arrow_shape)
                text_x = (x1 + x2) / 2
                text_y = (y1 + y2) / 2
                new_text = self.canvas.create_text(text_x, text_y, text=f"{weight:.2f}", fill="red")

                # Обновляем запись о связи
                conn_index = self.connections.index(conn)
                self.connections[conn_index] = (from_id, to_id, weight, new_line, new_text)

    def cancel_operation(self, event=None):
        if self.current_mode == "connect":
            if self.temp_line:
                self.canvas.delete(self.temp_line)
                self.temp_line = None
            self.connection_start = None
            self.current_mode = "select"
            self.status_var.set("Создание связи отменено")
        else:
            self.dragging = False
            self.drag_item = None
            self.selected_neuron = None
            self.status_var.set("Операция отменена")

    def add_neuron(self):
        x, y = 400, 300
        neuron_id = self.next_neuron_id
        self.next_neuron_id += 1

        circle = self.canvas.create_oval(x-20, y-20, x+20, y+20, fill="lightblue", tags=("neuron", str(neuron_id)))
        text = self.canvas.create_text(x, y, text=str(neuron_id), tags=("neuron_text", str(neuron_id)))

        self.neurons[neuron_id] = {
            "id": neuron_id,
            "name": f"Neuron {neuron_id}",
            "x": x,
            "y": y,
            "circle": circle,
            "text": text
        }

        self.status_var.set(f"Добавлен нейрон {neuron_id}")

    def rename_neuron(self):
        if not self.selected_neuron:
            messagebox.showwarning("Предупреждение", "Сначала выберите нейрон")
            return

        neuron_id = self.selected_neuron
        current_name = self.neurons[neuron_id]["name"]

        new_name = simpledialog.askstring("Изменить имя",
                                        f"Введите новое имя для нейрона {neuron_id}:",
                                        initialvalue=current_name)
        if new_name:
            self.neurons[neuron_id]["name"] = new_name
            x, y = self.neurons[neuron_id]["x"], self.neurons[neuron_id]["y"]
            self.canvas.delete(self.neurons[neuron_id]["text"])
            self.neurons[neuron_id]["text"] = self.canvas.create_text(
                x, y, text=f"{neuron_id}\n{new_name}", tags=("neuron_text", str(neuron_id)))

    def delete_element(self):
        if not self.selected_neuron:
            messagebox.showwarning("Предупреждение", "Сначала выберите нейрон для удаления")
            return

        neuron_id = self.selected_neuron

        # Удаляем связанные связи
        conns_to_remove = [conn for conn in self.connections if conn[0] == neuron_id or conn[1] == neuron_id]
        for conn in conns_to_remove:
            self.canvas.delete(conn[3])
            self.canvas.delete(conn[4])
            self.connections.remove(conn)

        # Удаляем нейрон
        self.canvas.delete(self.neurons[neuron_id]["circle"])
        self.canvas.delete(self.neurons[neuron_id]["text"])
        del self.neurons[neuron_id]

        self.selected_neuron = None
        self.status_var.set(f"Удален нейрон {neuron_id} и связанные связи")

    def show_matrix(self):
        self.update_weight_matrix()

        if self.weight_matrix is None:
            messagebox.showinfo("Матрица весов", "Нет данных для отображения матрицы")
            return

        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Матрица весов")

        text = tk.Text(matrix_window, width=40, height=20)
        text.pack(padx=10, pady=10)

        matrix_text = "Матрица весов (строка -> входы нейрона):\n\n"
        num_neurons = len(self.neurons)

        matrix_text += "    " + " ".join(f"{i+1:5}" for i in range(num_neurons)) + "\n"

        for i in range(num_neurons):
            neuron_id = i + 1
            if neuron_id not in self.neurons:
                continue
            matrix_text += f"{neuron_id:2} |" + " ".join(f"{w:5.2f}" if w is not None else "  None" for w in self.weight_matrix[i]) + "\n"

        text.insert(tk.END, matrix_text)
        text.config(state=tk.DISABLED)

    def update_weight_matrix(self):
        if not self.neurons:
            self.weight_matrix = None
            return

        num_neurons = len(self.neurons)
        self.weight_matrix = [[0]*num_neurons for _ in range(num_neurons)]

        for conn in self.connections:
            from_idx = conn[0] - 1
            to_idx = conn[1] - 1
            self.weight_matrix[to_idx][from_idx] = conn[2]

    def save_matrix(self):
        self.update_weight_matrix()

        if self.weight_matrix is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Сохранить матрицу весов")

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                num_neurons = len(self.neurons)
                f.write("Матрица весов (строка -> входы нейрона):\n\n")
                f.write("    " + " ".join(f"{i+1:5}" for i in range(num_neurons)) + "\n")

                for i in range(num_neurons):
                    neuron_id = i + 1
                    if neuron_id not in self.neurons:
                        continue
                    f.write(f"{neuron_id:2} |" + " ".join(f"{w:5.2f}" if w is not None else "  None" for w in self.weight_matrix[i]) + "\n")

            self.status_var.set(f"Матрица сохранена в {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def export_png(self):
        self.export_image("png")

    def export_svg(self):
        self.export_image("svg")

    def export_image(self, format):
        if not self.neurons:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return

        G = nx.DiGraph()

        for neuron_id, data in self.neurons.items():
            G.add_node(neuron_id, name=data["name"], pos=(data["x"], data["y"]))

        for conn in self.connections:
            G.add_edge(conn[0], conn[1], weight=conn[2])

        pos = {neuron_id: (data["x"], data["y"]) for neuron_id, data in self.neurons.items()}

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)

        labels = {neuron_id: f"{neuron_id}\n{data['name']}" for neuron_id, data in self.neurons.items()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        edge_labels = {(conn[0], conn[1]): f"{conn[2]:.2f}" for conn in self.connections}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_color='red')

        plt.axis('off')

        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{format}",
            filetypes=[(f"{format.upper()} files", f"*.{format}"), ("All files", "*.*")],
            title=f"Экспорт в {format.upper()}")

        if file_path:
            plt.savefig(file_path, format=format, bbox_inches='tight')
            plt.close()
            self.status_var.set(f"Сеть экспортирована в {file_path}")
        else:
            plt.close()

    def clear_all(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить все нейроны и связи?"):
            self.canvas.delete("all")
            self.neurons = {}
            self.connections = []
            self.weight_matrix = None
            self.next_neuron_id = 1
            self.selected_neuron = None
            self.status_var.set("Все данные удалены")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuronNetworkApp(root)
    root.mainloop()

