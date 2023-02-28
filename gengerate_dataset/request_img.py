import requests 	# 网络请求模块
import time			# 时间模块
import re 			# 正则表达式处理
import os			# 操作系统模块，调用系统命令

url = "http://cvh.bmicc.cn/cvh_server/tmp/CTRename/S_CTMR_CVH1_CT_cvh01_ct_0137.jpg"
# url = "http://cvh.bmicc.cn/cvh/cn/"
def download(http):
    # 请求网页前端代码 赋值给html
    html = requests.get(url=url)
    # 通过正则从代码中找到所有图片的链接，返回值是一个列表
    img_url = re.findall(r'http://.*\.jpg', html.text)
    # 循环这个列表
    for u in img_url:
        # 定义图片名字为链接的最后一段
        name = u.split('/')[-1]
        # 请求图片的链接，将图片的数据下载下来赋值给data
        data = requests.get(url=u).content
        # 创建储存图片的目录，无则创建
        img_dir = './image'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # 将图片数据写入到文件中
        with open(f'./image/{name}', 'wb') as f:
            f.write(data)
            print('图片', u, '下载完成')
        time.sleep(0.2)
    print('当前页面下载完毕，正在请求下一个 页面')
    time.sleep(2)
    # 用正则找到下一页的链接
    next_page = re.search(r'<a.*下一页.*</a>', html.text)
    next_page = next_page.group().split('</a><a')[-2]
    try:
        # 将函数返回值设置为下一页的链接
        return re.search(r'"(http.*?)"', next_page).group(1)
    except Exception:
        return None

def dowload_pic(url,mode):

    name = url.split('/')[-1]
    # 请求图片的链接，将图片的数据下载下来赋值给data
    data = requests.get(url=url).content
    # 创建储存图片的目录，无则创建
    img_dir = './image/%s/'%(mode)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # 将图片数据写入到文件中
    with open(os.path.join(img_dir,name), 'wb') as f:
        f.write(data)
        print('picture', "dowload from",url,"finish.")
    time.sleep(0.2)

for i in range(439,441):

    num = "%04d" % i

    # url = "http://cvh.bmicc.cn/cvh_server/tmp/slicesRename/S_CVH1_CVH01_%s.jpg"%(num)
    # url = "http://cvh.bmicc.cn/cvh_server/tmp/CTRename/S_CTMR_CVH1_CT_cvh01_ct_%s.jpg"%(num)
    url = "http://cvh.bmicc.cn/cvh_server/tmp/mriRename/S_CTMR_CVH1_MRI_cvh01_mri_%s.jpg"%(num)
    # print(url)
    # if i > 5 :
    #     break
    # dowload_pic(url,"silce")
    dowload_pic(url,"mri")
    time.sleep(0.5)