from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        min_temp, _ = torch.min(self.data,dim =1)
        max_temp, _ = torch.max(self.data,dim =1)
        return min_temp, max_temp

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """

        avg = torch.mean(self.data,dim =1 )
        d2d = avg[1:] - avg[:-1]

        return torch.min(d2d)

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        avg = torch.mean(self.data, dim =1, keepdim=True)
        diff_from_avg = torch.abs(self.data - avg)
        #gather get t
        extreme_temp = self.data.gather(1, torch.argmax(diff_from_avg, dim=1, keepdim=True))
        extreme_temp = extreme_temp.squeeze()
        return extreme_temp

        # to expand tensors to match dimesions we can use unsqueeze 
        # avg = avg.unsqueeze(1)
        # diff = torch.sub(self.data, avg)
        # max_diff, ind = torch.max(diff, dim =1)
        # extreme_measurements = self.data[torch.arange(self.data.size(0)), ind]
        # print(extreme_measurements)
        # return extreme_measurements
        
    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        t = self.data[-k:]
        m,_ = torch.max(t, dim =1)
        return m

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        temp = self.data[-k:]
        avg = torch.mean(temp)
        return avg

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        diff = torch.abs(self.data - t)
        avg = torch.mean(diff,dim =1)
        min_value, min_index = torch.min(avg, dim=0)
        return min_index



