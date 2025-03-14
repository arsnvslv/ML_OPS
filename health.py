import psutil
def check_system_health():
    # CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)

    # Memory usage
    memory_info = psutil.virtual_memory()

    # Disk usage
    disk_usage = psutil.disk_usage('/')

    return {
        'cpu_usage': f'{cpu_usage}%',
        'memory_usage': f'{memory_info.percent}%',
        'disk_usage': f'{disk_usage.percent}%'
    }
