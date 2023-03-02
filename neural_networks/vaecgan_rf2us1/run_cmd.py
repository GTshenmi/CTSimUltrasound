# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time

# cmd = 'python ~/hehe.py'


# def gpu_info():
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
#     #print(gpu_status)
#     gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
#     gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
#     return gpu_power, gpu_memory


import os
import psutil


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return used,free


# if __name__ == "__main__":
#
#     gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
#     print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
#           .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
#
#     cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
#     print(r'当前机器内存使用情况：总共 {} MB， 剩余 {} MB, 当前进程使用的内存 {} MB'
#           .format(cpu_mem_total, cpu_mem_free, cpu_mem_process_used))


def narrow_setup(cmd,gpu_id = 0,interval=2):
    used, free = get_gpu_mem_info(gpu_id)
    i = 0
    while free <= 7000:  # set waiting condition
        # print(used,free)
        used, free = get_gpu_mem_info(gpu_id)
        i = i % 5

        sys.stdout.write(f'gpu{gpu_id}:{used}/{used + free}\r\n')

        sys.stdout.flush()

        time.sleep(interval)

        i += 1
    #print('\n' + cmd)

    for c in cmd:
        os.system(c)


if __name__ == '__main__':

    python = "/home/xuepeng/miniconda3/envs/ultrasound/bin/python"
    exe_file = "/home/xuepeng/ultrasound/neural_networks/vaecgan_rf2us1/testmodel.py"
    log = "/home/xuepeng/ultrasound/neural_networks/vaecgan_rf2us1/logs/rf2us1_testmodel.log"

    cmd = [
        f'nohup {python} {exe_file} > {log} &'
    ]
    narrow_setup(cmd,1)
