# 根据关键词爬取 image.baidu.com 上的图片

import os.path
import requests
import time
from tqdm import tqdm
import urllib3
urllib3.disable_warnings()

cookies = {
    "BIDUPSID": "C9F91971EEEA5BC861C1515CA8C6F79F",
    "PSTM": "1682059189",
    "BAIDUID": "C9F91971EEEA5BC8669C495F7B53B830:FG=1",
    "BDORZ": "B490B5EBF6F3CD402E515D22BCDA1598",
    "H_PS_PSSID": "38516_36543_38686_38754_38767_38580_38680_38639_38766_26350_38568_38621",
    "BAIDUID_BFESS": "C9F91971EEEA5BC8669C495F7B53B830:FG=1",
    "delPer": "0",
    "PSINO": "1",
    "BA_HECTOR": "a4202001a50hag2g2g20a4cm1i7ge8a1n",
    "ZFY": "nunMIdr2UN9fcU4vyx9pGgiqXD0o9HbCrgdJWNMIz2U:C",
    "BDRCVFR[X_XKQks0S63]": "mk3SLVN4HKm",
    "userFrom": "null",
    "firstShowTip": "1",
    "indexPageSugList": "%5B%22%E8%A5%BF%E7%93%9C%22%5D",
    "cleanHistoryStatus": "0",
    "BDRCVFR[dG2JNJb_ajR]": "mk3SLVN4HKm",
    "BDRCVFR[-pGxjrCMryR]": "mk3SLVN4HKm",
    "ab_sr": "1.0.1_NDViOGVjZTQ5YmExYTUwMzIzYWNiOGNhNTVkYjJlYjJmNjg1YzE0NzcyM2MxMzFmNjM1OWViOTczNGVmZTc1ODU4YmRjNWMwYjNmZTk4YjkxYmIyYTdiNzZhMDU0NWYzZDAyYTU3ZDQ2MDE5NjFjZTQ2Yjk0Y2U4MWNjYThjOGMxNDRlN2U0NTc3NGQxMWE4NTM3OTgxZTVjNzE1ZTc5MA=="
}

headers = {
    # 'Accept': 'text/plain, */*; q=0.01',
    # 'Accept-Encoding': 'gzip, deflate, br',
    # 'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    # 'Connection': 'keep-alive',
    # 'Host': 'image.baidu.com',
    # 'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1685600884592_R&pv=&ic=0&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=&ie=utf-8&sid=&word=%E8%A5%BF%E7%93%9C',
    'sec-ch-ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
    # 'sec-ch-ua-mobile': '?0',
    # 'sec-ch-ua-platform': '"Linux"',
    # 'Sec-Fetch-Dest': 'empty',
    # 'Sec-Fetch-Mode': 'cors',
    # 'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    # 'X-Requested-With': 'XMLHttpRequest',
}


def craw_single_class(keyword, DOWNLOAD_NUM=20):
    if os.path.exists(f"../../datasets/custom_dataset/{keyword}"):
        print(f"../../datasets/custom_dataset/{keyword} 文件夹已存在")
    else:
        os.makedirs(f"../../datasets/custom_dataset/{keyword}")
        print(f"../../datasets/custom_dataset/{keyword} 文件夹已创建")

    count = 1
    Flag = True
    num = 0
    with tqdm(total=DOWNLOAD_NUM, position=0, leave=True) as pbar:
        while Flag:
            time.sleep(1)
            page = 30 * count

            # params in URL
            params = (
                ('tn', 'resultjson_com'),
                ('logid', '10958709967565886419'),
                ('ipn', 'rj'),
                ('ct', '201326592'),
                ('is', ''),
                ('fp', 'result'),
                ('fr', ''),
                ('word', f'{keyword}'),
                ('queryWord', f'{keyword}'),
                ('cl', '2'),
                ('lm', '-1'),
                ('ie', 'utf-8'),
                ('oe', 'utf-8'),
                ('adpicid', ''),
                ('st', '-1'),
                ('z', ''),
                ('ic', ''),
                ('hd', ''),
                ('latest', ''),
                ('copyright', ''),
                ('s', ''),
                ('se', ''),
                ('tab', ''),
                ('width', ''),
                ('height', ''),
                ('face', '0'),
                ('istype', '2'),
                ('qc', ''),
                ('nc', '1'),
                ('expermode', ''),
                ('nojc', ''),
                ('isAsync', ''),
                ('pn', f'{page}'),
                ('rn', '30'),
                ('gsm', '3c'),
                ('1647838001666', ''),
            )
            response = requests.get("https://image.baidu.com/search/acjson", headers=headers, params=params,
                                    cookies=cookies)
            if response.status_code == 200:
                try:
                    json_data = response.json().get("data")

                    if json_data:
                        for x in json_data:
                            type_ = x.get("type")
                            if type_ not in ["gif"]:
                                img = x.get("thumbURL")
                                fromPageTitleEnc = x.get("fromPageTitleEnc")
                                try:
                                    resp = requests.get(url=img, verify=False)
                                    time.sleep(1)
                                    file_save_path = os.path.join("../../datasets/custom_dataset", f"{keyword}",
                                                                  f"{str(num)}.{type_}")
                                    with open(file_save_path, "wb") as f:
                                        f.write(resp.content)
                                        f.flush()

                                        num += 1
                                        pbar.update(1)
                                    if num > DOWNLOAD_NUM:
                                        Flag = False
                                        print(f"{str(num)}张图像已经爬取完毕")
                                        break
                                except Exception:
                                    pass
                except:
                    pass
            else:
                break
            count += 1


if __name__ == "__main__":
    craw_single_class("西瓜")
    craw_single_class("杏")
    craw_single_class("栗子")
    craw_single_class("苹果")
