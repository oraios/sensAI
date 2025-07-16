from typing import List, Optional

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from matplotlib import pyplot as plt

from .pandas import SeriesInterpolationLinearIndex


class TensorboardData:
    def __init__(self, events: event_accumulator.EventAccumulator):
        self.events = events
        self.events.Reload()

    def get_series(self, tag: str, smoothing_factor: float = 0.0) -> pd.Series:
        """
        Gets the (smoothed) pandas Series for a specific tensorboard tag.

        :param tag: the tensorboard tag
        :param smoothing_factor: the smoothing factor between 0 and 1 which determines the relative importance of past values.
            0: no smoothing
            1: maximum smoothing (all values will be equal to the first value)
        :return: the pandas series with the step as the index
        """
        if not 0 <= smoothing_factor <= 1:
            raise ValueError("Smoothing factor must be between 0 and 1")

        try:
            scalar_events = self.events.Scalars(tag)
        except KeyError:
            raise KeyError(f"Tag '{tag}' not found in tensorboard events")

        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]

        if smoothing_factor > 0:
            smoothed_values = []
            last = values[0]
            for value in values:
                last = smoothing_factor * last + (1 - smoothing_factor) * value
                smoothed_values.append(last)
            values = smoothed_values

        return pd.Series(values, index=steps, name=tag)

    def get_tags(self) -> List[str]:
        """
        Get list of available scalar tags in the events.

        :return: list of tag names
        """
        return self.events.Tags()['scalars']

    def get_data_frame(self, tags: Optional[List[str]] = None, smoothing_factor: float = 0.0) -> pd.DataFrame:
        """
        Gets multiple series as a DataFrame.

        :param tags: the list of tensorboard tags to consider; if None, use all
        :param smoothing_factor: smoothing factor to apply to all series
        :return: DataFrame with steps as index and tags as columns
        """
        if tags is None:
            tags = self.get_tags()
        series_dict = {}
        for tag in tags:
            series = self.get_series(tag, smoothing_factor)
            series_dict[series.name] = series

        return pd.DataFrame(series_dict)


class TensorboardSeriesComparison:
    def __init__(self, tb_reference: TensorboardData, tb_current: TensorboardData,
            tag: str, index_start: int, index_end: int):
        s_ref = tb_reference.get_series(tag)
        s_cur = tb_current.get_series(tag)

        interp = SeriesInterpolationLinearIndex(ffill=True, bfill=True)
        s_ref, s_cur = interp.interpolate_all_with_combined_index([s_ref, s_cur])

        self.s_ref = s_ref.loc[index_start:index_end]
        self.s_cur = s_cur.loc[index_start:index_end]

    def mean_relative_difference(self):
        """
        Computes the difference between the current series and the reference series, relative to the reference,
        e.g. if the current series is on average 105% of the reference series (5% relative difference), then
        the value will be 0.05.
        Since we divide by the absolute value of the reference, this also works for negative cases, i.e.
        if the reference series value is -0.10 and the current series value is -0.08, then the relative
        difference is 0.2 (20%).

        :return: the mean relative difference
        """
        diff = self.s_cur - self.s_ref
        diff_rel = diff / abs(self.s_ref)
        return np.mean(diff_rel)

    def plot_series(self, show=False) -> plt.Figure:
        fig = plt.figure()
        self.s_ref.plot()
        self.s_cur.plot()
        plt.title(self.s_ref.name)
        if show:
            plt.show()
        return fig
