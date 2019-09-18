# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:55:07 2019

@author: yuxuan
"""

import os
import time
import json
import random
import socket
import requests
from selenium import webdriver
from urllib3.exceptions import MaxRetryError
from selenium.common.exceptions import TimeoutException

# write in txt
def writetxt(txtpath, contents):
    with open(txtpath, 'w') as fw:
        fw.write(contents)

# read data from file 
def readData(dataUrl):
    fopen=open(dataUrl, 'r')
    x=fopen.read()
    rawdata=x.split('}')
    listdict=[]
    for i in rawdata:
        try:
            i=i.lstrip(', ')
            i=i.lstrip('[')
            i=i.replace('}]','')
            i=i+'}'
            listdict.append(json.loads(i))
        except Exception: 
             continue 
    fopen.close()
    return listdict

# Recursively delete a folder
def delDirs(path_):
    path_ = path_.replace('\\', '/')
    if(os.path.isdir(path_)):
        for p in os.listdir(path_):
            delDirs(os.path.join(path_,p))
        if(os.path.exists(path_)):
            os.rmdir(path_)
    else:
        if(os.path.exists(path_)):
            os.remove(path_)

# format brand name
def brandName(brand):
    brand = brand.replace('/',',')
    brand = brand.replace('\\',',')
    brand = brand.replace(':',',')
    brand = brand.replace('*',',')
    brand = brand.replace('?',',')
    brand = brand.replace('<',',')
    brand = brand.replace('>',',')
    brand = brand.replace('"',',')
    brand = brand.replace('|',',')
    return brand

# get the data
def dataGet(url_target):
    socket.setdefaulttimeout(10) 
    okcount=0
    count200=0
    countsus=0
    countsock=0
    counttimeout=0
    countmax=0
    for i in url_target:  
        try:
            time_ = time.strftime("%Y-%m-%d-%H`%M`%S",time.localtime(time.time())) 
            brand = brandName(i['brand'])
            dirname = 'Database\\'+brand+'+'+time_  
            # start webdriver
            driver = webdriver.PhantomJS()
            driver.maximize_window()           
            driver.set_page_load_timeout(10)
            driver.set_script_timeout(10)
            url=i['url']
            if requests.get(url).status_code != 200:
                count200+=1
                driver.quit()
                continue
            driver.get(url)              
            content = driver.page_source
            if 'Account Suspended' in content:
                countsus+=1
                driver.quit()
                continue  
            if 'The website you were trying to reach is temporarily unavailable' in content:
                countsus+=1
                driver.quit()
                continue
            # make dirs
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # save screenshot and info  
            picpath = dirname+'\\shot.png'
            driver.save_screenshot(picpath)
            txtpath = dirname+'\\info.txt'
            writetxt(txtpath, str(i)) 
            htmlpath = dirname+'\\html.txt'           
            writetxt(htmlpath, content)
            # close driver_
            driver.quit()  
            print('ok') 
            okcount+=1
        except socket.timeout:
            countsock+=1
            driver.quit() 
            print('timeout')
            if os.path.exists(dirname):
                delDirs(dirname)   
        except TimeoutException:
            counttimeout+=1
            driver.quit() 
            print('timeout')
            if os.path.exists(dirname):
                delDirs(dirname)            
        except MaxRetryError:
            countmax+=1
            driver.quit()  
            print('err')               
            if os.path.exists(dirname):
                delDirs(dirname)    
        except Exception:
            driver.quit()  
            print('error')
            if os.path.exists(dirname):
                delDirs(dirname)
    print('ok:'+str(okcount))
    print('not 200:'+str(count200))
    print('suspand:'+str(countsus)) 
    print('socket'+str(countsock))
    print('timeout'+str(counttimeout))
    print('max'+str(countmax))
# the main function    
def main():
    dataUrl="Rawdata\\2019_7_22\\phishing_feed_30_days.txt"    
    url_target = readData(dataUrl)
    random.shuffle(url_target)
    urls = random.sample(url_target, 100)
    dataGet(urls)

if __name__ == '__main__': 
    main()