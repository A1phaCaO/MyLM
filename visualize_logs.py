import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# 修复负号显示
rcParams['axes.unicode_minus'] = False


def load_log_data(log_path):
    """
    加载日志文件数据
    
    Args:
        log_path (str): 日志文件路径
        
    Returns:
        dict: 日志数据
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_timestamp(timestamp):
    """
    解析时间戳为datetime对象或直接返回浮点数时间
    
    Args:
        timestamp: 时间戳（可以是字符串或浮点数）
        
    Returns:
        datetime或float: 解析后的时间对象或浮点数
    """
    # 如果是字符串，则解析为datetime对象
    if isinstance(timestamp, str):
        # 处理可能的微秒位数不一致问题
        if '.' in timestamp:
            base_part, frac_part = timestamp.split('.')
            frac_part = frac_part[:6].ljust(6, '0')  # 保证6位微秒
            timestamp = f"{base_part}.{frac_part}"
        return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    # 如果是数字（float或int），则直接返回
    else:
        return timestamp

def calculate_elapsed_time(timestamps):
    """
    计算相对于起始时间的经过时间（秒）
    
    Args:
        timestamps (list): 时间戳列表（可以是字符串或浮点数）
        
    Returns:
        list: 相对于起始时间的经过时间（秒）
    """
    # 处理混合类型的时间戳
    parsed_timestamps = [parse_timestamp(ts) for ts in timestamps]
    
    # 如果是datetime对象，则计算相对于起始时间的秒数
    if isinstance(parsed_timestamps[0], datetime):
        start_time = parsed_timestamps[0]
        return [(t - start_time).total_seconds() for t in parsed_timestamps]
    # 如果已经是浮点数（秒），则直接使用
    else:
        start_time = parsed_timestamps[0]
        return [t - start_time for t in parsed_timestamps]

def extract_training_data(log_data):
    """
    从日志数据中提取训练loss和时间信息
    
    Args:
        log_data (dict): 日志数据
        
    Returns:
        tuple: (elapsed_times, train_losses) 训练时间和对应的loss值
    """
    training_logs = log_data.get('training_logs', [])
    
    # 过滤出包含train_loss的记录（排除val_loss记录）
    train_logs = [log for log in training_logs if 'train_loss' in log]
    
    # 提取时间戳和loss值
    timestamps = [log.get('timestamp', i) for i, log in enumerate(train_logs)]  # 使用索引作为默认值
    train_losses = [log['train_loss'] for log in train_logs]
    
    # 计算相对于起始时间的经过时间
    elapsed_times = calculate_elapsed_time(timestamps)
    
    return elapsed_times, train_losses

def extract_validation_data(log_data):
    """
    从日志数据中提取验证loss和时间信息
    
    Args:
        log_data (dict): 日志数据
        
    Returns:
        tuple: (elapsed_times, val_losses) 训练时间和对应的验证loss值
    """
    training_logs = log_data.get('training_logs', [])
    
    # 过滤出包含val_loss的记录（排除train_loss记录）
    val_logs = [log for log in training_logs if 'val_loss' in log]
    
    # 提取时间戳和loss值
    timestamps = [log.get('timestamp', i) for i, log in enumerate(val_logs)]  # 使用索引作为默认值
    val_losses = [log['val_loss'] for log in val_logs]
    
    # 计算相对于起始时间的经过时间
    elapsed_times = calculate_elapsed_time(timestamps)
    
    return elapsed_times, val_losses

def create_label_from_log_data(log_data, log_file):
    """
    从日志数据中创建标签，使用model_class和备注信息
    
    Args:
        log_data (dict): 日志数据
        log_file (str): 日志文件路径
        
    Returns:
        str: 标签字符串
    """
    # 获取模型类名
    model_class = log_data.get('hyperparameters', {}).get('model_class', 'Unknown')
    
    # 获取备注信息
    notes = log_data.get('notes', '')
    if notes:
        return f"{model_class} ({notes})"
    else:
        return model_class

def plot_multiple_logs(log_files, labels=None, title="Training Loss vs Time",
                      figsize=(12, 8), save_path=None, use_val_loss=False):
    """
    在同一图表中绘制多个日志文件的loss-时间曲线
    
    Args:
        log_files (list): 日志文件路径列表
        labels (list, optional): 每条曲线的标签列表
        title (str): 图表标题
        figsize (tuple): 图表大小
        save_path (str, optional): 保存图片的路径
        use_val_loss (bool): 是否使用验证损失而不是训练损失
    """
    plt.figure(figsize=figsize)
     
    
    # 如果没有提供标签，则从日志数据中提取模型类名和备注作为标签
    if labels is None:
        labels = []
        for log_file in log_files:
            try:
                log_data = load_log_data(log_file)
                label = create_label_from_log_data(log_data, log_file)
                labels.append(label)
            except Exception as e:
                print(f"处理 {log_file} 时出错: {e}")
                # 使用文件名作为备选标签
                labels.append(Path(log_file).stem)
    
    # 获取matplotlib默认颜色循环
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 如果颜色不够，扩展颜色列表
    if len(colors) < len(log_files):
        # 通过调整亮度生成更多颜色
        import matplotlib.colors as mcolors
        base_colors = colors
        colors = []
        for i in range(len(log_files)):
            color_idx = i % len(base_colors)
            # 通过调整亮度生成新颜色
            base_color = mcolors.to_rgb(base_colors[color_idx])
            # 根据索引调整亮度
            factor = 1.0 - (i // len(base_colors)) * 0.3
            adjusted_color = tuple(max(0, min(1, c * factor)) for c in base_color)
            colors.append(adjusted_color)
    
    # 跟踪是否有任何数据被绘制
    has_data = False
    ylabel = 'Training Loss'  # 默认值
    
    # 为每条日志文件绘制曲线
    for i, log_file in enumerate(log_files):
        try:
            # 加载日志数据
            log_data = load_log_data(log_file)
            
            # 根据use_val_loss参数选择提取训练数据还是验证数据
            if use_val_loss:
                elapsed_times, losses = extract_validation_data(log_data)
                ylabel = 'Validation Loss'
            else:
                elapsed_times, losses = extract_training_data(log_data)
                ylabel = 'Training Loss'
            
            # 绘制曲线
            if elapsed_times and losses:
                plt.plot(elapsed_times, losses, label=labels[i], color=colors[i],
                        marker='o', markersize=4, linewidth=1.5)
                has_data = True
            
            print(f"成功加载 {log_file}: {len(elapsed_times)} 个数据点")
            
        except Exception as e:
            print(f"处理 {log_file} 时出错: {e}")
            continue
    
    # 设置图表属性
    plt.xlabel('训练时间 (秒)')
    plt.ylabel(ylabel)
    plt.title(title)
    
    # 只有在有数据时才显示图例
    if has_data:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def extract_step_validation_data(log_data):
    """
    从日志数据中提取按步数验证的loss和时间信息
    
    Args:
        log_data (dict): 日志数据
        
    Returns:
        tuple: (elapsed_times, steps, val_losses) 训练时间和对应的验证loss值
    """
    training_logs = log_data.get('training_logs', [])
    
    # 过滤出包含val_loss且类型为step_validation的记录
    val_logs = [log for log in training_logs 
                if 'val_loss' in log and log.get('type') == 'step_validation']
    
    # 提取时间戳、步骤和loss值
    timestamps = [log.get('timestamp', i) for i, log in enumerate(val_logs)]  # 使用索引作为默认值
    steps = [log['step'] for log in val_logs]
    val_losses = [log['val_loss'] for log in val_logs]
    
    # 计算相对于起始时间的经过时间
    elapsed_times = calculate_elapsed_time(timestamps)
    
    return elapsed_times, steps, val_losses

def plot_training_and_validation_losses(log_file, title=None, figsize=(12, 8),
                                       save_path=None, use_time=True):
    """
    在同一图表中绘制训练损失和验证损失曲线
    
    Args:
        log_file (str): 日志文件路径
        title (str, optional): 图表标题
        figsize (tuple): 图表大小
        save_path (str, optional): 保存图片的路径
        use_time (bool): True使用时间作为x轴，False使用步骤作为x轴
    """
    # 加载日志数据
    log_data = load_log_data(log_file)
    
    # 创建标签
    label = create_label_from_log_data(log_data, log_file)
    if not title:
        title = f"训练和验证损失曲线 - {label}"
    
    plt.figure(figsize=figsize)
    
    # 获取基础颜色
    base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    
    # 提取训练数据
    train_times, train_losses = extract_training_data(log_data)
    
    # 提取epoch验证数据
    epoch_val_times, epoch_val_losses = extract_validation_data(log_data)
    
    # 提取step验证数据
    step_val_times, steps, step_val_losses = extract_step_validation_data(log_data)
    
    # 跟踪是否有任何数据被绘制
    has_data = False
    
    if use_time:
        # 使用时间作为x轴
        if train_times and train_losses:
            plt.plot(train_times, train_losses, label='训练损失', color=base_color,
                    linestyle='-', alpha=0.7, linewidth=1.5)
            has_data = True
        if epoch_val_times and epoch_val_losses:
            plt.plot(epoch_val_times, epoch_val_losses, label='Epoch验证损失',
                    color=base_color, linestyle='--', marker='o', markersize=6, linewidth=1.5)
            has_data = True
        if step_val_times and step_val_losses:
            plt.plot(step_val_times, step_val_losses, label='Step验证损失',
                    color=base_color, linestyle='-.', marker='x', markersize=6, linewidth=1.5)
            has_data = True
        plt.xlabel('训练时间 (秒)')
    else:
        # 使用步骤作为x轴
        # 重新提取训练步骤数据
        training_logs = log_data.get('training_logs', [])
        train_logs = [log for log in training_logs if 'train_loss' in log]
        train_steps = [log.get('step', i) for i, log in enumerate(train_logs)]
        
        if train_steps and train_losses:
            plt.plot(train_steps, train_losses, label='训练损失', color=base_color,
                    linestyle='-', alpha=0.7, linewidth=1.5)
            has_data = True
            
        # 提取epoch验证步骤
        epoch_val_logs = [log for log in training_logs
                         if 'val_loss' in log and log.get('type') == 'epoch_validation']
        epoch_val_steps = [log.get('step', i) for i, log in enumerate(epoch_val_logs)]
        if epoch_val_steps and epoch_val_losses:
            plt.plot(epoch_val_steps, epoch_val_losses, label='Epoch验证损失',
                    color=base_color, linestyle='--', marker='o', markersize=6, linewidth=1.5)
            has_data = True
        
        if steps and step_val_losses:
            plt.plot(steps, step_val_losses, label='Step验证损失',
                    color=base_color, linestyle='-.', marker='x', markersize=6, linewidth=1.5)
            has_data = True
        plt.xlabel('训练步数')
    
    # 设置图表属性
    plt.ylabel('Loss')
    plt.title(title)
    
    # 只有在有数据时才显示图例
    if has_data:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def plot_multiple_logs_combined(log_files, labels=None, title="Training and Validation Loss Comparison",
                               figsize=(12, 8), save_path=None):
    """
    在同一图表中绘制多个日志文件的训练损失和验证损失曲线
    
    Args:
        log_files (list): 日志文件路径列表
        labels (list, optional): 每个日志文件的标签列表
        title (str): 图表标题
        figsize (tuple): 图表大小
        save_path (str, optional): 保存图片的路径
    # 重新应用中文字体设置，确保不被seaborn覆盖
    rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    """
    plt.figure(figsize=figsize)
    
    
    # 如果没有提供标签，则从日志数据中提取模型类名和备注作为标签
    if labels is None:
        labels = []
        for log_file in log_files:
            try:
                log_data = load_log_data(log_file)
                label = create_label_from_log_data(log_data, log_file)
                labels.append(label)
            except Exception as e:
                print(f"处理 {log_file} 时出错: {e}")
                # 使用文件名作为备选标签
                labels.append(Path(log_file).stem)
    
    # 获取matplotlib默认颜色循环
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 如果颜色不够，扩展颜色列表
    if len(colors) < len(log_files):
        # 通过调整亮度生成更多颜色
        import matplotlib.colors as mcolors
        base_colors = colors
        colors = []
        for i in range(len(log_files)):
            color_idx = i % len(base_colors)
            # 通过调整亮度生成新颜色
            base_color = mcolors.to_rgb(base_colors[color_idx])
            # 根据索引调整亮度
            factor = 1.0 - (i // len(base_colors)) * 0.3
            adjusted_color = tuple(max(0, min(1, c * factor)) for c in base_color)
            colors.append(adjusted_color)
    
    # 跟踪是否有任何数据被绘制
    has_data = False
    
    # 为每个日志文件绘制训练损失和验证损失曲线
    for i, log_file in enumerate(log_files):
        try:
            # 加载日志数据
            log_data = load_log_data(log_file)
            
            # 提取训练数据
            train_times, train_losses = extract_training_data(log_data)
            
            # 提取验证数据
            val_times, val_losses = extract_validation_data(log_data)
            
            # 使用相同的基础颜色，但不同的线型和标记
            base_color = colors[i]
            
            # 绘制训练损失曲线
            if train_times and train_losses:
                plt.plot(train_times, train_losses,
                        label=f'{labels[i]} - 训练损失',
                        color=base_color, linestyle='-', alpha=0.7, linewidth=1.5)
                has_data = True
            
            # 绘制验证损失曲线
            if val_times and val_losses:
                plt.plot(val_times, val_losses,
                        label=f'{labels[i]} - 验证损失',
                        color=base_color, linestyle='--', marker='o', markersize=4, linewidth=1.5)
                has_data = True
            
            print(f"成功加载 {log_file}: {len(train_times)} 个训练数据点, {len(val_times)} 个验证数据点")
            
        except Exception as e:
            print(f"处理 {log_file} 时出错: {e}")
            continue
    
    # 设置图表属性
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('Loss')
    plt.title(title)
    
    # 只有在有数据时才显示图例
    if has_data:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

# 示例用法
if __name__ == '__main__':
    
    # 在这里指定要可视化的日志文件路径
    log_files = [
        r'logs\LLaMABaseline(w- AdamW9e-3)_20251214_133702.json',
        # r'logs\LLaMABaseline(w- Muon9e-3)_20251214_145038.json',
        # r'logs\LLaMABaseline(w- Muon2d 9e-3)_20251214_150312.json',
        # r'logs\LLaMABaseline(w- Muon2dsync 9e-3)_20251214_151144.json',
        r'logs\LLaMABaseline(w- Muonsync 9e-3)_20251214_152251.json',
        r'logs\LLaMABaseline(w- Muonsync 5e-2)_20251214_153104.json',
        r'logs\LLaMABaseline_20251102_152549.json',
        r'logs\LLaMABaseline(w- Muonsync 1e-2)_20251214_153830.json',
        r'logs\LLaMABaseline(w- Muonsync 2.5e-3)_20251214_154420.json'
        # r'logs\LLaMABaseline(w- Muon(moonlight))_20251130_165809.json'
    ]
    
    # 调用新的绘图函数，同时显示训练损失和验证损失
    plot_multiple_logs_combined(
        log_files=log_files,
        labels=None,  # 设置为None以使用自动标签
        title="训练损失和验证损失对比",
        figsize=(12, 8),
        save_path=None  # 设置为文件路径以保存图片，例如："loss_curve.png"
    )
    
    # 保留原有的示例代码供参考
    # 可选：为每条曲线指定标签，如果为None则自动从日志中提取model_class和备注
    # labels = None
    # 
    # # 调用绘图函数，使用验证损失
    # plot_multiple_logs(
    #     log_files=log_files,
    #     labels=labels,  # 设置为None以使用自动标签
    #     title="验证Loss随时间变化曲线",
    #     figsize=(12, 8),
    #     save_path=None,  # 设置为文件路径以保存图片，例如："loss_curve.png"
    #     use_val_loss=True  # 使用验证损失
    # )
    # 
    # # 示例：绘制单个日志文件的训练和验证损失
    # plot_training_and_validation_losses(
    #     log_file=r'logs\MyLM_20250923_121547.json',
    #     title="训练和验证损失曲线",
    #     figsize=(12, 8),
    #     save_path=None,
    #     use_time=True
    # )
