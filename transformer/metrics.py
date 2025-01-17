from dataclasses import dataclass
from typing import Callable, Dict, List, Literal

@dataclass
class Metric:
    """
    The Metric object class.

    Args:
        - `calculate_metric`: The function to calculate the metric.
            The function should take two parameters:
            - The first one for predictions (List of Strings).
            - The second for references (List of Strings).
            
            The function should return a numerical value representing the metric.
        
        - `decode`: True for decoding sequences before calculating the metric, False otherwise.
        - `best_value_criteria`: 'min' if the metric is considered better when minimized, 'max' if it's considered better when maximized.
    """
    calculate_metric: Callable
    decode: bool = False
    best_value_criteria: Literal['min', 'max'] = 'max'

class MetricsEvaluator:
    """
    Wrapper class to calculate evaluation metrics.
    New metrics can be added and existing metrics can be updated.
    """
    def __init__(self, tokenizer = None, metrics: Dict[str, Metric] = {}) -> None:
        """
        Creates a new MetricsEvaluator.

        Args:
            - `tokenizer`: The tokenizer that will be used to decode the predictions and references with token ids.
            - `metrics`: A dict with the metric name as key and a Metric object as value.
        """
        self.tokenizer = tokenizer
        self.metrics = metrics
        self._any_decode = any(metric_settings.decode for metric_settings in self.metrics.values())

    def add_metric(self, name: str, metric: Metric) -> None:
        """
        Adds or updates a metric.

        Args:
            - `name`: The name of the metric.
            - `metric`: The Metric object to be added.
        """
        self.metrics[name] = metric
        self._any_decode = any(metric_settings.decode for metric_settings in self.metrics.values())

    def get_metric(self, name: str) -> Metric:
        return self.metrics.get(name, None)

    @staticmethod
    def metric_value_is_better(curr_value: int|float, comparison_value: int|float, best_value_criteria: Literal['min', 'max']) -> bool:
        """
        Returns True if `comparison_value` is better than `curr_value`, returns False otherwise.
        The best metric value is obtained based on `best_value_criteria`, which could be 'min' for finding the minimum or 'max' for obtaining the maximum.
        """
        if best_value_criteria == 'min':
            return comparison_value < curr_value
        return comparison_value > curr_value

    def __call__(self, predictions: List[List[int]], references: List[List[int]]) -> Dict[str, int|float]:
        if self._any_decode and self.tokenizer is None:
            raise ValueError('There is no tokenizer to decode the sequences')

        if self._any_decode and self.tokenizer is not None:
            # Decode batch returns List[str]
            decoded_preds = self.tokenizer.decode_batch(predictions, skip_special_tokens=True)
            decoded_refs = self.tokenizer.decode_batch(references, skip_special_tokens=True)

        metric_values = {}
        for name, metric in self.metrics.items():
            try:
                selected_preds, selected_refs = (decoded_preds, decoded_refs) if metric.decode else (predictions, references)
                metric_values[name] = metric.calculate_metric(selected_preds, selected_refs)
            except Exception as e:
                print(f'Error while calculating the metric {name}: {e}')
                raise

        return metric_values
