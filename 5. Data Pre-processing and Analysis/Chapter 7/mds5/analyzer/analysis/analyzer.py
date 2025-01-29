import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols

warnings.filterwarnings('ignore')


class TTTH_Analyzer:

    def __init__(self):
        pass

    def analyze_category_variable(self, variable_name, df):
        """
        This function support to analyze category variable by count values and show bar plot of variable count
        parameter variable_name: name of categorical column
        parameter df: DataFrame include categorical column
        """
        class_count = self._count_values_of_variable(variable_name, df)
        print(f'Class count of {variable_name}:\n')
        print('==========')
        print(f'{class_count}')
        print('==========')
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        class_count.head(7).plot.bar()
        plt.title(f'Bar chart of {variable_name}')
        plt.show()

    def show_central_tendency(self, variable_name, df):
        """
        Show central tendency of continuous variable
        parameter: variable_name: name of continuous variable
        parameter df: DataFrame include continuous column
        """
        _mean = df[variable_name].mean()
        _median = df[variable_name].median()
        _mode = df[variable_name].mode()
        _min = df[variable_name].min()
        _max = df[variable_name].max()
        _range = _max - _min
        result = {'mean': _mean, 'median': _median, 'mode': _mode[0], 'min': _min, 'max': _max, 'range': _range}
        print(f'central tendency of {variable_name}: {result}')

    def show_dispersion(self, variable_name, df):
        """
        Show dispersion of continuous variable
        parameter: variable_name: name of continuous variable
        parameter df: DataFrame include continuous column
        """
        _min = df[variable_name].min()
        _max = df[variable_name].max()
        _range = _max - _min
        q1 = df[variable_name].quantile(0.25)
        q3 = df[variable_name].quantile(0.75)
        iqr = q3 - q1
        var = df[variable_name].var()
        skew = df[variable_name].skew()
        kurtosis = df[variable_name].kurtosis()
        result = {'range': _range, 'q1': q1, 'q3': q3, 'iqr': iqr, 'var': var, 'skew': skew, 'kurtosis': kurtosis}
        print(f'Dispersion of {variable_name}: \n {result}')

    def visualize_hist_box_plot(self, variable_name, df):
        """
        Show histogram plot of continuous variable
        parameter: variable_name: name of continuous variable
        parameter df: DataFrame include continuous column
        """
        data = df[variable_name]
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        data.plot.hist()
        plt.title(f'Hist plot of {variable_name}')
        plt.subplot(1, 2, 2)
        data.plot.box()
        plt.title(f'Box plot of {variable_name}')
        plt.show()

    def analyze_numeric_variable(self, variable_name, df):
        """
        This function support to analyze numerical variable by central tendency, dispersion
         and show hist plot of continuous variable
        parameter: variable_name: name of continuous variable
        parameter df: DataFrame include continuous column
        """
        print('=====')
        self.show_central_tendency(variable_name, df)
        print('=====')
        self.show_dispersion(variable_name, df)
        print('=====')
        self.visualize_hist_box_plot(variable_name, df)

    @staticmethod
    def create_unique_pair_variable(variables):
        """
        Create unique pair of category variables
        Ex: [{A,B}, {A,C}, {B,C}]
        parameter variables: list of category variable
        return: list of unique pair of category variable
        """
        unique_pair = []
        for col in variables[:-1]:
            for _col in variables[1:]:
                if (col == _col) or ({col, _col} in unique_pair):
                    continue
                else:
                    unique_pair.append({col, _col})
        return unique_pair

    def use_chi_2_evaluation(self, tw_table, prob=0.95):
        """
        Use chi2 to check 2 category in two-way table Dependent or not
        parameter tw_table: two-way table of 2 category variable
        parameter prob: percent of chance that accept null hypothesis (H0-2 variable independent)
        """
        stats, p, dof, expected = chi2_contingency(tw_table)
        alpha = 1 - prob
        if p <= alpha:
            return 'Reject H0 - Dependent'
        else:
            return 'Fail to reject H0 - Independent'

    def create_tw_table(self, var1, var2, df):
        """
        Create two-way table base on DataFrame and 2 category input
        parameter var1: name of category 1
        parameter var2: name of category 2
        parameter df: DataFrame include 2 category variables input
        """
        tw_table = pd.crosstab(df[var1], df[var2])
        return tw_table

    def analyze_category_vs_category(self, var1, var2, df, prob=0.95):
        """
        Create two-way table, show stacked bar and use chi2 to evaluate 2 category variable dependent or not
        parameter var1: name of category 1
        parameter var2: name of category 2
        parameter df: DataFrame include 2 category variables input
        """
        tw_table = self.create_tw_table(var1, var2, df)
        print(f'=====Analyze of {var1} and {var2}=====')
        tw_table.plot(kind='bar', stacked=True, legend=False)
        chi2_result = self.use_chi_2_evaluation(tw_table, prob)
        plt.title(chi2_result)
        plt.show()
        return {'var1': var1, 'var2': var2, 'result': chi2_result}

    def visualize_box_for_continous_vs_categories(self, continous_var, category_vars, df):
        """
        Visualize boxplot for continuous and 1 or 2 category variables
        parameter continous_var: continuous variable
        parameter category_vars: str for 1 variable or list for 2 category variables
        parameter df: DataFrame contains continuous and category variables input
        """
        if isinstance(category_vars, str):
            sns.boxplot(x=category_vars, y=continous_var, data=df)
        elif isinstance(category_vars, list) and (len(category_vars) == 2):
            sns.boxplot(x=category_vars[0], y=continous_var, hue=category_vars[1], data=df)
        else:
            raise ValueError('Only support for 2 categories variable analysis')
        plt.legend([], [], frameon=False)
        plt.xticks(rotation=90)
        plt.show()

    def analyze_anova_table_for_continous_vs_categories(self, continous_var, category_vars, df):
        """
        Use ANOVA table to analyze continuous vs 1 or 2 category variables
        parameter continous_var: continuous variable
        parameter category_vars: str for 1 variable or list for 2 category variables
        parameter df: DataFrame contains continuous and category variables input
        """
        if isinstance(category_vars, str):
            function = f'{continous_var} ~ C({category_vars})'
        elif isinstance(category_vars, list) and (len(category_vars) == 2):
            function = f'{continous_var} ~C({category_vars[0]}) + C({category_vars[1]}) + C({category_vars[0]}):C({category_vars[1]})'
        else:
            raise ValueError('Only support for 2 categories variable analysis')
        model = ols(function, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

    def analyze_continous_vs_categories(self, continous_var, category_vars, df):
        """
        Analyze continous variables with 1 or 2 category variables
        parameter continous_var: continuous variable
        parameter category_vars: str for 1 variable or list for 2 category variables
        parameter df: DataFrame contains continuous and category variables input
        """
        self.analyze_anova_table_for_continous_vs_categories(continous_var, category_vars, df)
        print('======')
        self.visualize_box_for_continous_vs_categories(continous_var, category_vars, df)

    @staticmethod
    def _count_values_of_variable(variable_name, df, normalize=False):
        """
        Count sub categories of category variable
        :param df: DataFrame
        :param variable_name: category variable name
        :param normalize: convert to ratio or not
        :return:
        """
        df_category = df[variable_name]
        class_count = df_category.value_counts(normalize=normalize)
        return class_count

    def check_imbalance_class(self, variable_name, df):
        """
        Check ratio of each class
        :param variable_name: Variable name
        :param df: DataFrame contains category
        :return:
        """
        class_count = self._count_values_of_variable(variable_name, df, normalize=True)
        print(f'Class count of {variable_name}:\n')
        print('==========')
        print(f'{class_count}')
        print('==========')
        max_class = class_count.max()
        min_class = class_count.min()
        ratio_of_classes = max_class / min_class
        print(f'Ratio of 2 class is {round(ratio_of_classes, 2)}')
        if ratio_of_classes >= 2:
            print(f'You should consider to handle imbalance')

    def check_outlier_of_numerical_variable(self, numerical_variable, df):
        """
        Function check outlier of variable and return index of outlier if any
        :param numerical_variable: Name of Numerical variable
        :param df: DataFrame contains Numerical variable
        :return: Index of upper and lower outlier
        """
        total_sample = df.shape[0]
        q1 = df[numerical_variable].quantile(0.25)
        q3 = df[numerical_variable].quantile(0.75)
        iqr = q3 - q1
        limit_up = q3 + 1.5 * iqr
        limit_low = q1 - 1.5 * iqr
        index_up = df[df[numerical_variable] > limit_up].index
        index_low = df[df[numerical_variable] < limit_low].index
        upper_outlier_ratio = index_up.shape[0] / total_sample
        lower_outlier_ratio = index_low.shape[0] / total_sample
        if index_up.empty and index_low.empty:
            print(f'Variable {numerical_variable} have no outlier')
            return
        else:
            print(f'variable {numerical_variable} have {round(upper_outlier_ratio * 100, 3)}% upper outlier')
            print(f'variable {numerical_variable} have {round(lower_outlier_ratio * 100, 3)}% lower outlier')
            return index_up, index_low
