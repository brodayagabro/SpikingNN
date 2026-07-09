import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import os
from collections import defaultdict


class TableMerger:
    """
    Класс для анализа и объединения таблиц.
    
    При объединении:
    - Сохраняются ВСЕ уникальные колонки из всех файлов
    - Пропущенные значения заполняются NaN (null)
    - Порядок колонок: сначала общие, затем уникальные в алфавитном порядке
    """
    
    def __init__(self):
        self.dataframes = {}
        self.column_structures = {}
        self.file_paths = {}
        self.errors = {}
    
    def load_csv_files(self, file_paths: List[str], encoding: str = 'utf-8') -> None:
        """
        Загрузка CSV файлов из массива путей
        
        Args:
            file_paths: Список путей к файлам (могут быть в разных папках)
            encoding: Кодировка файлов
        """
        print(f"📁 Загрузка {len(file_paths)} файлов...\n")
        
        for path in file_paths:
            try:
                file_path = Path(path)
                
                if not file_path.exists():
                    self.errors[path] = "Файл не существует"
                    print(f"❌ {path} - файл не найден")
                    continue
                
                if not file_path.is_file():
                    self.errors[path] = "Это не файл"
                    print(f"❌ {path} - это не файл")
                    continue
                
                # Чтение с основной кодировкой
                df = pd.read_csv(file_path, encoding=encoding)
                
                file_name = file_path.name
                unique_key = f"{file_name}_{len(self.dataframes)}"
                
                self.dataframes[unique_key] = df
                self.column_structures[unique_key] = list(df.columns)
                self.file_paths[unique_key] = str(file_path)
                
                print(f"✅ {file_name} ({df.shape[0]} строк, {df.shape[1]} колонок)")
                
            except PermissionError as e:
                self.errors[path] = f"Нет доступа: {e}"
                print(f"🔒 {path} - нет прав доступа")
            except UnicodeDecodeError:
                # Пробуем альтернативные кодировки
                for alt_encoding in ['cp1251', 'latin-1', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(file_path, encoding=alt_encoding)
                        file_name = file_path.name
                        unique_key = f"{file_name}_{len(self.dataframes)}"
                        
                        self.dataframes[unique_key] = df
                        self.column_structures[unique_key] = list(df.columns)
                        self.file_paths[unique_key] = str(file_path)
                        
                        print(f"✅ {file_name} (кодировка: {alt_encoding})")
                        break
                    except:
                        continue
                else:
                    self.errors[path] = "Не удалось определить кодировку"
                    print(f"❌ {path} - ошибка кодировки")
            except Exception as e:
                self.errors[path] = str(e)
                print(f"❌ {path} - {type(e).__name__}: {e}")
        
        print(f"\n📊 Загружено: {len(self.dataframes)} файлов")
        if self.errors:
            print(f"⚠️ Ошибки: {len(self.errors)} файлов")
    
    def get_all_unique_columns(self, file_keys: List[str]) -> List[str]:
        """
        Получение всех уникальных колонок из указанных файлов
        
        Возвращает колонки в порядке: сначала общие для всех, затем уникальные (алфавитно)
        """
        if not file_keys:
            return []
        
        # Собираем все колонки
        all_columns = set()
        column_sets = []
        
        for key in file_keys:
            if key in self.column_structures:
                cols = set(self.column_structures[key])
                all_columns.update(cols)
                column_sets.append(cols)
        
        if not column_sets:
            return []
        
        # Находим общие колонки (пересечение всех множеств)
        common_columns = set.intersection(*column_sets) if len(column_sets) > 1 else column_sets[0]
        
        # Уникальные колонки (не общие)
        unique_columns = all_columns - common_columns
        
        # Формируем итоговый порядок: сначала общие (в порядке первого файла), потом уникальные (алфавитно)
        first_file_cols = self.column_structures[file_keys[0]] if file_keys[0] in self.column_structures else []
        
        # Общие колонки в порядке первого файла
        ordered_common = [c for c in first_file_cols if c in common_columns]
        # Добавляем общие колонки, которых не было в первом файле
        ordered_common += sorted([c for c in common_columns if c not in ordered_common])
        
        # Уникальные колонки в алфавитном порядке
        ordered_unique = sorted(unique_columns)
        
        return ordered_common + ordered_unique
    
    def merge_files(self, file_keys: List[str], 
                    ignore_index: bool = True,
                    add_source_column: bool = False,
                    fill_value: any = None) -> pd.DataFrame:
        """
        Объединение файлов с сохранением ВСЕХ колонок.
        
        Все колонки из всех файлов будут в результате.
        Пропущенные значения заполняются fill_value (по умолчанию NaN/null).
        
        Args:
            file_keys: Список ключей файлов для объединения
            ignore_index: Сбросить индексы
            add_source_column: Добавить колонку с именем источника
            fill_value: Значение для заполнения пропусков (None = NaN)
        
        Returns:
            Объединённый DataFrame со всеми колонками
        """
        if not file_keys:
            raise ValueError("Список файлов пуст")
        
        available_keys = [k for k in file_keys if k in self.dataframes]
        
        if not available_keys:
            raise ValueError("Нет доступных файлов для объединения")
        
        if len(available_keys) == 1:
            df = self.dataframes[available_keys[0]].copy()
            if add_source_column:
                df['_source_file'] = Path(self.file_paths[available_keys[0]]).name
            return df
        
        # Получаем все уникальные колонки для правильного порядка
        all_columns = self.get_all_unique_columns(available_keys)
        
        dfs = []
        for key in available_keys:
            df = self.dataframes[key].copy()
            
            if add_source_column:
                df['_source_file'] = Path(self.file_paths[key]).name
            
            # Приводим к единому набору колонок (добавляем отсутствующие с NaN)
            for col in all_columns:
                if col not in df.columns:
                    df[col] = fill_value  # NaN по умолчанию
            
            # Приводим порядок колонок к единому
            df = df[all_columns + (['_source_file'] if add_source_column else [])]
            dfs.append(df)
        
        # Объединяем с outer join (по умолчанию) - сохраняет все колонки
        merged = pd.concat(dfs, ignore_index=ignore_index, sort=False)
        
        # Явно заполняем пропуски если указано
        if fill_value is not None and pd.isna(fill_value) == False:
            merged = merged.fillna(fill_value)
        
        return merged
    
    def find_mergeable_groups(self) -> Dict[str, List[str]]:
        """Поиск групп файлов по идентичной структуре колонок"""
        groups = defaultdict(list)
        
        for file_key, columns in self.column_structures.items():
            # Сигнатура: отсортированный кортеж колонок
            signature = tuple(sorted(columns))
            groups[signature].append(file_key)
        
        return dict(groups)
    
    def compare_all_structures(self) -> Dict:
        """Подробное сравнение структур всех файлов"""
        if not self.column_structures:
            return {'error': 'Нет загруженных файлов'}
        
        groups = self.find_mergeable_groups()
        
        # Эталон: первая группа или первый файл
        reference_key = list(self.column_structures.keys())[0]
        reference_columns = self.column_structures[reference_key]
        
        # Все уникальные колонки
        all_unique = set()
        for cols in self.column_structures.values():
            all_unique.update(cols)
        
        result = {
            'total_files': len(self.dataframes),
            'total_groups': len(groups),
            'reference_columns': reference_columns,
            'all_unique_columns': sorted(all_unique),
            'mergeable_groups': groups,
            'errors': self.errors,
            'column_statistics': self._get_column_statistics()
        }
        
        return result
    
    def _get_column_statistics(self) -> Dict[str, Dict]:
        """Статистика по колонкам: в каких файлах встречаются"""
        stats = {}
        
        for key, columns in self.column_structures.items():
            for col in columns:
                if col not in stats:
                    stats[col] = {'count': 0, 'files': []}
                stats[col]['count'] += 1
                stats[col]['files'].append(Path(self.file_paths[key]).name)
        
        return stats
    
    def merge_all_with_all_columns(self, add_source_column: bool = False,
                                   fill_value: any = None) -> Optional[pd.DataFrame]:
        """
        Объединение ВСЕХ загруженных файлов.
        
        Результат содержит ВСЕ уникальные колонки из всех файлов.
        Пропущенные значения заполняются fill_value (по умолчанию NaN).
        """
        if not self.dataframes:
            print("❌ Нет файлов для объединения")
            return None
        
        file_keys = list(self.dataframes.keys())
        
        print(f"🔗 Объединяем {len(file_keys)} файлов...")
        print(f"📊 Будет сохранено всех уникальных колонок: {len(self.get_all_unique_columns(file_keys))}")
        
        merged = self.merge_files(
            file_keys=file_keys,
            add_source_column=add_source_column,
            fill_value=fill_value
        )
        
        print(f"✅ Результат: {merged.shape[0]} строк, {merged.shape[1]} колонок")
        return merged
    
    def print_report(self, comparison: Dict) -> None:
        """Вывод подробного отчёта о структурах"""
        print("\n" + "="*80)
        print("📋 ОТЧЁТ ПО СТРУКТУРЕ ТАБЛИЦ")
        print("="*80)
        
        print(f"\n📊 Всего файлов: {comparison['total_files']}")
        print(f"📊 Всего групп с идентичной структурой: {comparison['total_groups']}")
        print(f"📊 Всего уникальных колонок во всех файлах: {len(comparison['all_unique_columns'])}")
        
        print(f"\n📌 Все уникальные колонки:")
        for i, col in enumerate(comparison['all_unique_columns'], 1):
            stat = comparison['column_statistics'].get(col, {})
            count = stat.get('count', 0)
            marker = "⭐" if count == comparison['total_files'] else "◦"
            print(f"   {marker} {i}. {col} (в {count}/{comparison['total_files']} файлах)")
        
        print(f"\n📁 Группы с идентичной структурой:")
        for i, (signature, files) in enumerate(comparison['mergeable_groups'].items(), 1):
            status = "✅" if len(files) >= 2 else "⚠️"
            print(f"\n   {status} Группа {i} ({len(files)} файлов):")
            for f in files:
                rows = self.dataframes[f].shape[0]
                path = self.file_paths[f]
                print(f"      • {Path(path).name} ({rows} строк)")
        
        if comparison['errors']:
            print(f"\n❌ Ошибки загрузки ({len(comparison['errors'])}):")
            for path, error in comparison['errors'].items():
                print(f"   • {Path(path).name}: {error}")
    
    def save_merged(self, df: pd.DataFrame, output_path: str, 
                    na_representation: str = '') -> bool:
        """
        Сохранение объединённой таблицы
        
        Args:
            df: DataFrame для сохранения
            output_path: Путь к выходному файлу
            na_representation: Строка для представления NaN в CSV (по умолчанию пустая строка)
        """
        if df is None or df.empty:
            print("❌ Нечего сохранять")
            return False
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем с указанием представления для NaN
            df.to_csv(output_path, index=False, encoding='utf-8-sig', na_rep=na_representation)
            
            print(f"💾 Файл сохранён: {output_path}")
            print(f"📊 Размер: {df.shape[0]} строк × {df.shape[1]} колонок")
            
            # Статистика пропусков
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                print(f"⚠️ Пропущенных значений: {null_counts.sum()}")
                print("   Колонки с пропусками:")
                for col, count in null_counts[null_counts > 0].items():
                    pct = count / len(df) * 100
                    print(f"     • {col}: {count} ({pct:.1f}%)")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения: {type(e).__name__}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Полная статистика по всем файлам"""
        stats = {
            'total_files': len(self.dataframes),
            'total_rows': sum(df.shape[0] for df in self.dataframes.values()),
            'all_columns': self.get_all_unique_columns(list(self.dataframes.keys())),
            'files_info': []
        }
        
        for key, df in self.dataframes.items():
            file_info = {
                'key': key,
                'path': self.file_paths[key],
                'name': Path(self.file_paths[key]).name,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': list(df.columns),
                'null_count': int(df.isnull().sum().sum()),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
            stats['files_info'].append(file_info)
        
        return stats


# ==================== УДОБНЫЕ ФУНКЦИИ ====================

def merge_csv_files_universal(file_paths: List[str], 
                              output_path: str,
                              add_source_column: bool = False,
                              encoding: str = 'utf-8',
                              fill_null_with: any = None,
                              na_representation: str = '') -> Optional[pd.DataFrame]:
    """
    Универсальное объединение CSV файлов.
    
    Все колонки из всех файлов сохраняются в результате.
    Пропущенные значения заполняются null (NaN) или fill_null_with.
    
    Args:
        file_paths: Список путей к файлам
        output_path: Путь для сохранения результата
        add_source_column: Добавить колонку '_source_file' с именем файла
        encoding: Кодировка входных файлов
        fill_null_with: Значение для заполнения пропусков (None = оставить NaN)
        na_representation: Как записать NaN в CSV файл (по умолчанию пустая строка)
    
    Returns:
        Объединённый DataFrame или None при ошибке
    """
    merger = TableMerger()
    merger.load_csv_files(file_paths, encoding=encoding)
    
    if not merger.dataframes:
        print("❌ Нет файлов для обработки")
        return None
    
    comparison = merger.compare_all_structures()
    merger.print_report(comparison)
    
    # Объединяем все файлы с сохранением всех колонок
    merged = merger.merge_all_with_all_columns(
        add_source_column=add_source_column,
        fill_value=fill_null_with
    )
    
    if merged is not None:
        merger.save_merged(merged, output_path, na_representation=na_representation)
        return merged
    
    return None


def analyze_csv_structures(file_paths: List[str], encoding: str = 'utf-8') -> Dict:
    """Только анализ структур без объединения"""
    merger = TableMerger()
    merger.load_csv_files(file_paths, encoding=encoding)
    
    if not merger.dataframes:
        return {'error': 'Нет загруженных файлов'}
    
    return merger.compare_all_structures()


# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================

def example_merge_different_structures():
    """Пример: Объединение файлов с разными колонками"""
    print("\n" + "="*80)
    print("ПРИМЕР: Файлы с разной структурой → все колонки в одном файле")
    print("="*80)
    
    # Файлы с разными наборами колонок
    file_paths = [
        r'C:\data\experiment1.csv',      # колонки: [time, voltage, current]
        r'C:\data\experiment2.csv',      # колонки: [time, voltage, frequency]
        r'C:\data\experiment3.csv',      # колонки: [time, current, amplitude, phase]
    ]
    
    # Результат будет иметь колонки: [time, voltage, current, frequency, amplitude, phase]
    # В строках из experiment1: frequency, amplitude, phase = NaN
    # В строках из experiment2: current, amplitude, phase = NaN
    # и т.д.
    
    df = merge_csv_files_universal(
        file_paths=file_paths,
        output_path=r'C:\output\merged_all_columns.csv',
        add_source_column=True,
        fill_null_with=None,  # Оставить NaN (можно заменить на 0, -1, 'N/A' и т.д.)
        na_representation=''  # Пустая строка для NaN в выходном CSV
    )
    
    if df is not None:
        print(f"\n✅ Успешно! Объединено {len(df)} строк с {len(df.columns)} колонками")
        print(f"\n📋 Колонки в результате:")
        for i, col in enumerate(df.columns, 1):
            null_pct = df[col].isnull().sum() / len(df) * 100
            print(f"   {i}. {col} ({null_pct:.1f}% пропусков)")


def example_fill_nulls_with_value():
    """Пример: Заполнение пропусков конкретным значением"""
    print("\n" + "="*80)
    print("ПРИМЕР: Заполнение пропусков значением по умолчанию")
    print("="*80)
    
    file_paths = [
        r'path/to/file1.csv',
        r'path/to/file2.csv',
    ]
    
    # Заполнить все пропуски значением 0
    df = merge_csv_files_universal(
        file_paths=file_paths,
        output_path=r'output/filled_with_zero.csv',
        fill_null_with=0  # Все отсутствующие значения будут 0
    )
    
    # Или заполнить строкой для текстовых колонок
    # df = merge_csv_files_universal(..., fill_null_with='N/A')


# ==================== БЫСТРЫЙ ЗАПУСК ====================

if __name__ == "__main__":
    # ==================== НАСТРОЙКИ ====================
    
    # 1. Массив путей к вашим файлам (исправлены пропущенные запятые)
    file_paths = [
            r'./New_experiment_results_const_current_without_afferents.csv',
            r'./New_experiment_results_meandr_pulses_with_afferents.csv',
            r'./New_experiment_results_sinusoidal_currents_with_afferents.csv',
            r'./New_experiment_results_sin_without_afferents.csv',
            r'./New_experiment_results_const_current_with_afferents.csv',
            r'./New_experiment_results_meandr_pulse_without_afferents.csv',
    ]
    
    # 2. Запуск объединения
    merged_df = merge_csv_files_universal(
        file_paths=file_paths,
        output_path='./all_merged_experiments.csv',
        add_source_column=True,      # Добавить колонку с именем файла-источника
        encoding='utf-8',            # Кодировка входных файлов
        fill_null_with=None,         # None = оставить NaN, или укажите 0, -999, 'N/A' и т.д.
        na_representation=''         # Как записать NaN в CSV: '' = пустая строка
    )
    
    # 3. Работа с результатом
    if merged_df is not None:
        print("\n" + "="*80)
        print("📊 ИНФОРМАЦИЯ ОБ ОБЪЕДИНЁННОЙ ТАБЛИЦЕ")
        print("="*80)
        
        print(f"\n📈 Размеры: {merged_df.shape[0]} строк × {merged_df.shape[1]} колонок")
        
        print(f"\n📋 Все колонки:")
        for i, col in enumerate(merged_df.columns, 1):
            null_count = merged_df[col].isnull().sum()
            null_pct = null_count / len(merged_df) * 100
            marker = "✓" if null_count == 0 else f"⚠ {null_pct:.1f}%"
            print(f"   {i:2d}. {col:30s} [{marker}]")
        
        print(f"\n🔍 Первые 5 строк:")
        #print(merged_df.head().to_string())
        
        # Сохранить также в Excel для удобства просмотра
        try:
            excel_path = Path(r'./merged_all_experiments.xlsx')
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_excel(excel_path, index=False)
            print(f"\n💾 Также сохранено в Excel: {excel_path}")
        except Exception as e:
            print(f"\n⚠️ Не удалось сохранить Excel: {e}")
