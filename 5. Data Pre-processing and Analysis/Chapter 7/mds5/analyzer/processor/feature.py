import warnings

warnings.filterwarnings('ignore')


class FeatureProcessor:

    def __init__(self):
        pass

    @staticmethod
    def handle_missing_values_by_mode(variable_name, df):
        """This function use to replace missing values of category variable by mode

        :param variable_name: category variable
        :param df: dataframe include category variable
        :return:
        """
        print(f'{variable_name} before fill missing values: {df[variable_name].isna().sum()}')
        mode = df[variable_name].mode()[0]
        missing_index = df[df[variable_name].isna()].index
        df.loc[missing_index, variable_name] = mode
        print(f'{variable_name} after fill missing values: {df[variable_name].isna().sum()}')

    @staticmethod
    def handle_missing_values_by_median(variable_name, df):
        """This function use to replace missing values of numeric variable by median

        :param variable_name: numeric variable
        :param df: dataframe include numeric variable
        :return:
        """
        print(f'{variable_name} before fill missing values: {df[variable_name].isna().sum()}')
        median = df[variable_name].median()
        missing_index = df[df[variable_name].isna()].index
        df.loc[missing_index, variable_name] = median
        print(f'{variable_name} after fill missing values: {df[variable_name].isna().sum()}')

    @staticmethod
    def handle_uncommon_category(variable_name, df, threshold=10, label='Rare'):
        """This function group uncommon category which count lower than threshold to label input

        :param variable_name: category variable to group uncommon category
        :param df: dataframe include category variable
        :param threshold: count limit to define uncommon category: Default is 10
        :param label: Label after group uncommon category
        :return:
        """
        count_values = df[variable_name].value_counts()
        print(f'Before group uncommon category to {label} \n: {count_values}')
        rare_category = count_values[count_values < threshold].index
        rare_index = df[df[variable_name].isin(rare_category)].index
        df.loc[rare_index, variable_name] = label
        re_count = df[variable_name].value_counts()
        print(f'After group uncommon category to {label} \n: {re_count}')
