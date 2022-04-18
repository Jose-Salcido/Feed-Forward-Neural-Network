from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn


class PlottingFunctions:
    def __init__(
        self,
        output_path: Path,
        dataframe,
        history,
        plot_label,
        x_truth_data,
        y_test,
        y_predictions
    ) -> None:
        self.output_path = output_path
        self.dataframe = dataframe
        self.history = history
        self.plot_label = plot_label
        self.truth_data = x_truth_data
        self.y_test = y_test
        self.y_predictions = y_predictions

    def Histogram(self):
        bins = 100
        plt.figure(figsize = (15, 10))
        plt.title(f"Histogram Plot: Output Data Distribution (bins = {bins})", fontsize = 20, weight = "bold")
        plt.ylabel("Counts", fontsize = 14)
        plt.xlabel("Output Data Value", fontsize = 20)
        plt.hist(self.truth_data, bins = bins, label = self.plot_label, alpha = 0.8)
        plt.legend(fontsize = 20)
        plt.savefig(self.output_path / Path("Histogram_Output_Distribution.png"))

    def CorrHeatMap(self):
        corr = self.dataframe.corr()
        size = len(self.dataframe.columns)
        fig, ax = plt.subplots(figsize=(4 * size, 4 * size))  # Dynamically change size of plot
        ax.set_title("Data Correlation Heat Map", fontsize = 20, weight = "bold")
        colormap = seaborn.diverging_palette(220, 10, as_cmap = True)
        seaborn.heatmap(corr, cmap = colormap, annot = True, fmt = ".4f")
        ax.set_yticklabels(labels = corr.columns, va = 'center')
        ax.set_xticklabels(labels = corr.columns, ha = 'center')
        plt.savefig(self.output_path / Path('Data_Correlation_HeatMap.png'))

    def LossPlot(self):
        fig, ax = plt.subplots(figsize=(20, 15))

        plt.title("Loss", fontsize = 20, weight = "bold")
        plt.ylabel("Loss", fontsize = 20)
        plt.xlabel("Epoch", fontsize = 20)
        loss = np.min(self.history.history['loss'])
        plt.plot(self.history.history['loss'], label = "train")
        plt.plot(self.history.history["val_loss"], label = "test")
        plt.legend(fontsize = 20)
        plt.text(0.05, 0.05, f'Min Train Loss: {loss:.5f}',
                 ha = 'left', va = 'bottom', fontsize = 12, transform = ax.transAxes)
        plt.savefig(self.output_path / Path("Loss.png"))
        return loss

    def MSEPlot(self):
        plt.figure(figsize = (15, 10))
        plt.title("Mean Squared Error", fontsize = 20, weight = "bold")
        plt.ylabel("MSE", fontsize = 20)
        plt.xlabel("Epoch", fontsize = 20)
        plt.plot(self.history.history["mse"], label = "train")
        plt.plot(self.history.history["val_mse"], label = "test")
        plt.legend(fontsize=20)
        plt.savefig(self.output_path / Path("Mean_Squared_Error.png"))

    def MAEPlot(self):
        plt.figure(figsize = (15, 10))
        plt.title("Mean Absolute Error", fontsize = 20, weight = 'bold')
        plt.ylabel("MAE", fontsize = 20)
        plt.xlabel("Epoch", fontsize = 20)
        plt.plot(self.history.history["mae"], label = "train")
        plt.plot(self.history.history["val_mae"], label = "test")
        plt.legend(fontsize = 20)
        plt.savefig(self.output_path / Path("Mean_Absolute_Error.png"))

    def LogLogPlot(self):
        x = self.y_test
        y = self.y_predictions
        plt.figure(figsize=(15, 10))
        plt.title("Log-Log Plot: Predicted vs. Truth", fontsize=20, weight='bold')
        plt.ylabel(f"Predicted {self.plot_label}", fontsize=20)
        plt.xlabel(f"Truth {self.plot_label}", fontsize=20)
        plt.loglog(x, y, 'o', markersize=4, alpha=0.7)
        plt.savefig(self.output_path / Path("LogLogPlot_Truth_vs_Predicted.png"))

    def ScatterPlot(
        self,
        plot_stats: bool = True
    ):
        fig, ax = plt.subplots(figsize=(20, 15))
        x = self.y_test
        y = self.y_predictions
        # Calculate the regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[:, 0], y[:, 0])
        yhat = intercept + slope * x
        plt.title("Scatter Plot: Predicted vs. Truth", fontsize = 20, weight = "bold")
        plt.ylabel(f"Predicted {self.plot_label}", fontsize = 20)
        plt.xlabel(f"Truth {self.plot_label}", fontsize = 20)
        plt.scatter(x, y, s = 5, alpha = 0.7)
        plt.plot(x, yhat, color = 'g', label = 'regression line')  # Over-plot the regression line in green
        plt.plot((np.min(y), np.max(x)), (np.min(y), np.max(x)), 'r-',
                 label = 'truth line')  # Over-plot the truth line in red
        r2 = r_value ** 2

        if plot_stats:
            # Plot statistics calculated from stats.lingregress
            plt.text(0.9, 0.1, f'Standard Error : {std_err:.5f}',
                     ha = 'right', va = 'bottom', fontsize = 12, transform = ax.transAxes)
            plt.text(0.9, 0.08, f'R-squared : {r2:.5f}',
                     ha = 'right', va = 'bottom', fontsize = 12, transform = ax.transAxes)
            plt.text(0.9, 0.06, f'P-Val : {p_value:.5f}',
                     ha = 'right', va = 'bottom', fontsize = 12, transform = ax.transAxes)

        plt.legend(fontsize = 20)
        plt.savefig(self.output_path / Path('ScatterPlot_Truth_vs_Predicted.png'))

        return x, y, slope, intercept, r2, p_value

    def ResidualPlot(self):
        x = self.y_test
        y = self.y_predictions - self.y_test

        # Calculate the regression line with scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[:, 0], y[:, 0])
        yhat = intercept + slope * x
        plt.figure(figsize = (15, 10))
        plt.title("Scatter Plot: Residual vs. Truth", fontsize = 20, weight = "bold")
        plt.ylabel("Residual (prediction - truth)", fontsize = 20)
        plt.xlabel(f"Truth {self.plot_label}", fontsize = 20)
        plt.scatter(x, y, s = 5, alpha = 0.7)
        plt.plot(x, yhat, color = 'k', label = 'regression line')  # Over-plot the regression line in green
        plt.legend(fontsize = 20)
        plt.savefig(self.output_path / Path("ScatterPlot_Residual_vs_Truth.png"))

    def cdf_histogram_raw_error(self):
        raw_error = self.y_predictions - self.y_test

        # Determine min/max range for bins
        min_bin = np.floor(raw_error.min())
        max_bin = np.ceil(raw_error.max())
        bin_count = int(max_bin - min_bin)

        # Generate figure and ax1 which will hold the histogram
        fig, ax1 = plt.subplots(figsize = (20, 15))
        ax1.hist(raw_error, bins = bin_count)
        ax1.set_xlabel('Raw Error', fontsize = 18)
        ax1.set_ylabel("Counts", fontsize = 18)
        ax1.set_title('Histogram-CDF Raw Error', fontsize = 20, weight = 'bold', pad = 20)

        # Use the same x-axis as ax1
        ax2 = ax1.twinx()
        ax2.set_ylabel("Cumulative Sum", fontsize=18)

        count, bin_edges = np.histogram(raw_error, bins=bin_count)
        cdf = np.cumsum(count)
        ax2.plot(bin_edges[1:], cdf, color = 'tab:orange', label = 'Cumulative Sum', linewidth = 3)

        # Plot the legend on the upper left corner of the figure
        fig.legend(loc = "upper left", bbox_to_anchor = (0, 1), bbox_transform = ax1.transAxes, fontsize = 15)
        # Save the figure as png file
        fig.savefig(self.output_path / Path('Histogram-CDF Raw Error.png'))

    def cdf_histogram_percent_error(
        self,
        plot_mean: bool = True
    ):

        percent_error = (abs(self.y_predictions - self.y_test) / abs(self.y_test)) * 100

        # Determine min/max range for bins
        min_bin = np.floor(percent_error.min())
        max_bin = np.ceil(percent_error.max())
        bin_count = int(max_bin - min_bin)

        # Generate figure and ax1 which will hold the histogram
        fig, ax1 = plt.subplots(figsize = (20, 15))
        ax1.hist(percent_error, bins = bin_count, range = (percent_error.min(), 100.0))
        ax1.set_xlabel('Percent Error (%)', fontsize = 18)
        ax1.set_ylabel('Counts', fontsize = 18, color = 'tab:blue')
        ax1.set_title('Histogram-CDF Percent Error', fontsize=20, weight='bold', pad=20)

        # Draw quantile line
        if plot_mean:
            mean_error = np.mean(percent_error)
            ax1.axvline(x = mean_error, color = 'red', label = f"Mean Error: {mean_error:.2f}%")
            ax1.text(mean_error, -10, "{:.2f}".format(mean_error), color = "red", ha = "right", va = "center")

        # Use the same x-axis as ax1
        ax2 = ax1.twinx()
        ax2.set_ylabel("Cumulative Sum", fontsize = 18, color = 'tab:orange')

        # Generate histogram, compute CDF and save figure
        count, bin_edges = np.histogram(percent_error, bins=bin_count, range=(percent_error.min(), 100.0))
        cdf = np.cumsum(count)
        ax2.plot(bin_edges[1:], cdf, color = 'tab:orange', label = 'Cumulative Sum', linewidth = 3)
        fig.legend(loc = "upper left", bbox_to_anchor = (0, 1), bbox_transform = ax1.transAxes, fontsize = 15)
        fig.savefig(self.output_path / Path('Histogram-CDF Percent Error.png'))
