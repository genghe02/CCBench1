"""
测试retry能力
"""
import pandas as pd


def original_error_type(retry_log_path, model_name):
    # 分析初始错误类型（占比、饼状图or雷达图？）
    import pandas as pd
    data = pd.read_json(retry_log_path)

    data['model'] = [model_name] * len(data)

    initial_errors = data[['task_id', 'error']]

    error_mapping = {
        # 'Success': 0,
        'time out': 1,
        'test_points': 2,
        'others': 3
    }
    initial_errors['error_code'] = initial_errors['error'].map(error_mapping)

    error_counts = initial_errors.groupby('error').size().reset_index(name='count')

    # 计算总任务数
    total_tasks = len(initial_errors)

    # 计算错误类型出现比例
    error_counts['error_rate'] = error_counts['count'] / total_tasks

    print(error_counts)


def retry_success_analisys(json_file, model_name):
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ========== 1. 读取日志并转换为DataFrame ==========

    def load_retry_logs(json_file_path):
        """
        从给定路径加载单个模型的retry日志，并转换为pandas DataFrame。
        返回的DataFrame可以用来做后续分析。
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for entry in data:
            task_id = entry.get('task_id', None)
            initial_error = entry.get('error', None)
            retry_time = entry.get('retry_time', 0)
            error_type_list = entry.get('error_type', [])

            # 检查是否出现success
            # 注意：有的日志中可能是'Success'大小写不同，也可能成功后就终止了
            # 这里做一个统一的小写判断
            success_flag = any(e.lower() == 'success' for e in error_type_list)

            final_status = 'success' if success_flag else 'fail'

            records.append({
                'task_id': task_id,
                'initial_error': initial_error,
                'retry_time': retry_time,
                'error_type_list': error_type_list,
                'final_status': final_status
            })

        df = pd.DataFrame(records)
        return df

    # ========== 2. 示例分析 ==========

    def analyze_retry_success_rate(df):
        """
        对单个模型的DataFrame进行统计，包括成功率、平均retry次数等。
        打印结果并返回一个分析结果字典。
        """
        total_tasks = len(df)
        success_tasks = len(df[df['final_status'] == 'success'])
        fail_tasks = total_tasks - success_tasks
        success_rate = success_tasks / total_tasks if total_tasks > 0 else 0.0

        # 平均retry次数：分两个口径
        avg_retry_all = df['retry_time'].mean() if total_tasks > 0 else 0
        avg_retry_success = df[df['final_status'] == 'success']['retry_time'].mean() \
            if success_tasks > 0 else 0

        print("=== Retry Success Rate Analysis ===")
        print(f"Total tasks: {total_tasks}")
        print(f"Success tasks: {success_tasks}")
        print(f"Fail tasks: {fail_tasks}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average retry (all tasks): {avg_retry_all:.2f}")
        print(f"Average retry (only success tasks): {avg_retry_success:.2f}")

        # 返回一个字典，便于后续可视化
        return {
            "total_tasks": total_tasks,
            "success_tasks": success_tasks,
            "fail_tasks": fail_tasks,
            "success_rate": success_rate,
            "avg_retry_all": avg_retry_all,
            "avg_retry_success": avg_retry_success
        }

    # ========== 3. 可视化示例 ==========

    def plot_retry_distributions(df, model_name="MyModel"):
        """
        可视化retry次数分布，以及成功/失败的数量对比。
        """
        # 设置Seaborn风格
        sns.set(style="whitegrid")

        # 3.1 成功 vs 失败 数量
        plt.figure(figsize=(6, 4))
        status_counts = df['final_status'].value_counts()
        sns.barplot(x=status_counts.index, y=status_counts.values, palette="Set2")
        plt.title(f"{model_name} - Success vs Fail Count")
        plt.xlabel("Final Status")
        plt.ylabel("Count")
        plt.show()

        # 3.2 retry_time分布（区分成功 vs 失败）
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='final_status', y='retry_time', data=df, palette="Set2")
        plt.title(f"{model_name} - Retry Times by Final Status")
        plt.xlabel("Final Status")
        plt.ylabel("Retry Times")
        plt.show()

        # 3.3 如果想看retry_time的直方图
        plt.figure(figsize=(6, 4))
        sns.histplot(df['retry_time'], bins=range(0, 12), kde=True, color='cornflowerblue')
        plt.title(f"{model_name} - Distribution of Retry Times (All Tasks)")
        plt.xlabel("Retry Times")
        plt.ylabel("Frequency")
        plt.show()


    # 1) 读取日志
    df_model = load_retry_logs(json_file)

    # 2) 计算并打印分析结果
    analysis_result = analyze_retry_success_rate(df_model)

    # 3) 可视化
    plot_retry_distributions(df_model, model_name=model_name)

def retry_single_task_id_analysis(json_file, model_name):
    def load_retry_logs(json_file_path):
        import json
        """
        从给定路径加载单个模型的retry日志，并转换为pandas DataFrame。
        返回的DataFrame可以用来做后续分析。
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for entry in data:
            task_id = entry.get('task_id', None)
            initial_error = entry.get('error', None)
            retry_time = entry.get('retry_time', 0)
            error_type_list = entry.get('error_type', [])

            # 检查是否出现success
            # 注意：有的日志中可能是'Success'大小写不同，也可能成功后就终止了
            # 这里做一个统一的小写判断
            success_flag = any(e.lower() == 'success' for e in error_type_list)

            final_status = 'success' if success_flag else 'fail'

            records.append({
                'task_id': task_id,
                'initial_error': initial_error,
                'retry_time': retry_time,
                'error_type_list': error_type_list,
                'final_status': final_status
            })

        df = pd.DataFrame(records)
        return df

    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 500)
    def analyze_by_task_id(df):
        """
        按 task_id 分组，对每个 task_id 上的多条记录进行统计。
        这里的思路：把同一个task_id的所有记录分别算作一次独立测试。
        输出：
          - counts: 每个 task_id 一共测试了几次
          - success_count: 在这几次测试中，成功了多少次
          - fail_count: 失败了多少次
          - success_rate: success_count / counts
          - avg_retry: 在这几次测试中的平均 retry 次数
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', 500)
        grouped = df.groupby('task_id')

        results = []
        for task, group in grouped:
            total_count = len(group)
            success_count = sum(group['final_status'] == 'success')
            fail_count = total_count - success_count
            success_rate = success_count / total_count if total_count > 0 else 0
            avg_retry = group['retry_time'].mean()

            results.append({
                'task_id': task,
                'num_records': total_count,
                'success_count': success_count,
                'fail_count': fail_count,
                'success_rate': success_rate,
                'avg_retry_time': avg_retry
            })

        df_task_summary = pd.DataFrame(results)
        return df_task_summary

    import seaborn as sns
    import matplotlib.pyplot as plt

    def plot_task_id_performance(df_task_summary):
        # 按 success_rate 排序，画条形图
        df_sorted = df_task_summary.sort_values(by='success_rate', ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_sorted, x='task_id', y='success_rate', palette='Blues_d')
        plt.xticks(rotation=45)
        plt.title("Success Rate by Task ID")
        plt.tight_layout()
        plt.show()

        # 画平均retry时间
        df_sorted = df_task_summary.sort_values(by='avg_retry_time', ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_sorted, x='task_id', y='avg_retry_time', palette='Oranges_d')
        plt.xticks(rotation=45)
        plt.title("Average Retry Time by Task ID")
        plt.tight_layout()
        plt.show()

    def analyze_by_task_id_once_success(df):
        """
        对于每个 task_id，如果在多条记录中至少有一次 final_status == success，
        则认为这个task最终可以被成功解决。否则视为失败。

        输出：
          - total_count: 参与统计的task数量
          - success_count: 有成功记录的task数量
          - fail_count: 没有一次成功的task数量
          - success_rate: success_count / total_count
          - avg_retry_time_for_tasks_that_eventually_succeed
            => 统计只要成功过的task，取该task所有成功记录的平均retry次数（或最小值）
            => 具体如何定义要看需求
        """
        grouped = df.groupby('task_id')

        task_ids = []
        success_flags = []
        min_retry_times_for_success = []
        avg_retry_times_for_success = []

        for task, group in grouped:
            # 是否有成功记录
            has_success = (group['final_status'] == 'success').any()
            success_flags.append(has_success)
            task_ids.append(task)

            if has_success:
                # 取所有成功记录中最小的 retry_time 或平均值
                success_rows = group[group['final_status'] == 'success']
                min_retry_time = success_rows['retry_time'].min()
                avg_retry_time = success_rows['retry_time'].mean()
            else:
                min_retry_time = None
                avg_retry_time = None

            min_retry_times_for_success.append(min_retry_time)
            avg_retry_times_for_success.append(avg_retry_time)

        results_df = pd.DataFrame({
            'task_id': task_ids,
            'final_success': success_flags,
            'min_retry_if_success': min_retry_times_for_success,
            'avg_retry_if_success': avg_retry_times_for_success
        })

        # 统计指标
        total_tasks = len(results_df)
        success_tasks = sum(results_df['final_success'] == True)
        fail_tasks = total_tasks - success_tasks
        success_rate = success_tasks / total_tasks if total_tasks > 0 else 0.0

        print("=== Task-level (once success) Analysis ===")
        print(f"Total tasks: {total_tasks}")
        print(f"Tasks that had at least one success: {success_tasks}")
        print(f"Tasks with zero success: {fail_tasks}")
        print(f"Success rate: {success_rate:.2%}")

        # 你可以也对min_retry_if_success做一些统计，比如平均值、中位数等
        successful_tasks_df = results_df[results_df['final_success'] == True]
        mean_min_retry = successful_tasks_df['min_retry_if_success'].mean()
        mean_avg_retry = successful_tasks_df['avg_retry_if_success'].mean()

        print(f"Mean of min_retry_if_success: {mean_min_retry:.2f}")
        print(f"Mean of avg_retry_if_success: {mean_avg_retry:.2f}")

        return results_df

    # 使用示例


    df_model = load_retry_logs(json_file)
    # 使用示例
    df_task_summary = analyze_by_task_id(df_model)

    print(df_task_summary)

    plot_task_id_performance(df_task_summary)

    results_df_task_once_success = analyze_by_task_id_once_success(df_model)
    print(results_df_task_once_success)
    import pandas

    results_df_task_once_success.to_csv(f"./retry_results/{model_name}.csv")


if __name__ == '__main__':
    # Set model
    # model_name = 'gpt4o-mini'
    # model_name = 'deepseek_v2'
    model_name = 'gpt3_5'
    assert model_name in ['deepseek_v2', 'gpt3_5', 'gpt4o-mini', 'mistral',  "deepseek_7b"]

    # Set retry_log_path
    if model_name == 'deepseek_v2':
        retry_log_path = "deepseek_v2_retry_log.json"
    elif model_name == 'gpt3_5':
        retry_log_path = "gpt3_5_retry_log.json"
    elif model_name == 'gpt4o-mini':
        retry_log_path = "gpt4o-mini_retry_log.json"
    elif model_name == 'mistral':
        retry_log_path = "mistral_retry_log.json"
    elif model_name == "deepseek_7b":
        retry_log_path = "deepseek_7b_retry_log.json"


    """
    初始错误分析
    """

    # original_error_type(retry_log_path, model_name)

    """
    retry后成功率分析
    """

    # retry_success_analisys(retry_log_path, model_name)

    """
    单个模型在每个task_id上的表现
    """

    # retry_single_task_id_analysis(retry_log_path, model_name)

    """
    不同模型对于样本的表现
    是否对于难的样本不同模型都表现不好
    还是对于样本而言，不同模型的表现不同（解决不了的样本不是同一批）
    家族内-跨家族 对于问题的表现
    """

    sample_dict = {}

    for _i, model_name in enumerate(['deepseek_v2', 'gpt3_5', 'gpt4o-mini']):
        df = pd.read_csv(f"./retry_results/{model_name}.csv", header=0)
        for i in range(df.shape[0]):
            if df.iloc[i][1] not in sample_dict.keys():
                sample_dict[df.iloc[i][1]] = [-1, -1, -1]
            sample_dict[df.iloc[i][1]][_i] = df.iloc[i][4]

    pd.DataFrame(sample_dict).to_csv(f"./retry_results/1.csv", index=False)


