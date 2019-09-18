# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:08:07 2019

@author: syxuan
"""

import os
import re
import cv2
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib3.exceptions import MaxRetryError
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# Get the urls
def urlsGet():
    readfile="web_urls.txt"
    f=open(readfile, "r", encoding="utf8")
    lines=f.readlines()    
    i=0
    targets=[]
    urls=[]  
    for line in lines:    
        i+=1;
        if i%2==0:
            urls.append(line)
        else:
            targets.append(line.strip('\n'))
    return urls, targets

# String splite
def findStr(string, subStr, findCnt):
    listStr = string.split(subStr,findCnt)
    if len(listStr) <= findCnt:
        return -1
    return len(string)-len(listStr[-1])-len(subStr)

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
            
# Target tag can't be the child of 'head' tag
def header(link):
    for parent in link.parents:
        if parent is None:
            break
        elif parent.name=='head' or parent.name=='noscript':
            return True
    return False

# Get the info of the input box
def inputGet(driver, soup, tag):
    flag = True
    links = driver.find_elements_by_xpath("//input")
    inputs = soup.find_all('input')
    for k in range(0,len(links)):
        try:
            size_=links[k].size
        except:
            continue
        if(size_['width']!=0 and size_['height']!=0):
            if (str(inputs[k]).find('search') != -1 or str(inputs[k]).find('Search')!=-1):
                continue
            if (str(inputs[k]).find('user') != -1 or str(inputs[k]).find('email')!=-1 
                 or str(inputs[k]).find('e-mail')!=-1 or str(inputs[k]).find('login')!=-1
                 or str(inputs[k]).find('Email')!=-1 or str(inputs[k]).find('text')!=-1) :
                flag = False
                break
    return flag, links[k].rect

# Get the info of the logo location
def logoGet(url, driver, soup, tag):
    flag = True
    s = findStr(url, '/', 3)
    originurl = url[0:s]+'/'
    
    links1 = driver.find_elements_by_xpath("//img")   
    links2 = driver.find_elements_by_xpath("//*[name()='svg']")   
    links3 = driver.find_elements_by_xpath("//*[contains(@id,'logo') or contains(@id,'Logo')]")
    links4 = driver.find_elements_by_xpath("//*[contains(@class,'logo') or contains(@class,'Logo')]")        
    links5_ = driver.find_elements_by_xpath("//a[@href]")
    links6 = driver.find_elements_by_xpath("//a")  
    links5=[]
    for link in links5_:
        if link.get_attribute("href")==originurl:
            links5.append(link)   
    links = links1 + links2 + links3 + links4 + links5 + links6
    
    logos1 = soup.find_all(name='img')
    logos2 = soup.find_all(name='svg')
    logos3 = soup.find_all(id=re.compile('logo',flags=re.IGNORECASE))
    logos4 = soup.find_all(class_=re.compile('logo',flags=re.IGNORECASE))
    logos5=[]
    logos6 = soup.find_all(name='a')
    for logo_ in soup.find_all('a', attrs={'href':True }):
        if logo_['href']=='/' or logo_['href']==originurl:
            logos5.append(logo_)   
    logos=[]             
    logos_ = logos1 + logos2 + logos3 + logos4 + logos5 + logos6
    for logo in logos_:
        if header(logo):
            continue
        logos.append(logo)
    
    for k in range(0,len(links)):
        try:
            size_=links[k].size
        except:
            continue
        if(size_['width']!=0 and size_['height']!=0):
            if (re.compile('logo',flags=re.IGNORECASE).findall(str(logos[k])) != [] or
                ('href='+originurl) in str(logos[k]) or
                'href="/"' in str(logos[k])) :  
                flag = False
                break   
    return flag, links[k].rect

# Draw the bounding box
def boundingBox(k, path, img, inputbox, logobox): 
    # Input
    inputpx1 = (int(inputbox['x']-10), int(inputbox['y']-10))
    inputpx2 = (int(inputbox['x']+inputbox['width']+10), int(inputbox['y']+inputbox['height']+10))
    cv2.rectangle(img, inputpx1, inputpx2, (0,0,255), 2)
    cv2.putText(img, 'input', inputpx1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    inputpath = path + '\\input_mark.txt'
    f_new = open(inputpath, 'w')
    f_new.write(str(inputbox))
    f_new.close()
    
    # Logo
    logopx1 = (int(logobox['x']-10), int(logobox['y']-10))
    logopx2 = (int(logobox['x']+logobox['width']+10), int(logobox['y']+logobox['height']+10))
    cv2.rectangle(img, logopx1, logopx2, (0,0,255), 2)
    cv2.putText(img, 'logo', logopx1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)    
    
    logopath = path + '\\logo_mark.txt'
    f_new = open(logopath, 'w')
    f_new.write(str(logobox))
    f_new.close()
    
    # Draw the bounding_box picture
    boundpath = path+'\\bounding.png'
    cv2.imwrite(boundpath, img)
#    cv2.imwrite('D:\\yuxuan\\trainSet\\11bound\\'+str(k)+'.png', img)

# Main function
def main():
    urls, targets = urlsGet()    
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400',
        'referer':'http://www.taobao.com' }
    cap = DesiredCapabilities.PHANTOMJS.copy() 
    for key, value in headers.items():
        cap['phantomjs.page.customHeaders.{}'.format(key)] = value     
        
    for k in range(0, len(urls)):
        try:
            print(str(k))
            url=urls[k]          
            tag=targets[k]
            driver = webdriver.PhantomJS()
            driver.set_page_load_timeout(10)
            driver.set_script_timeout(10)
            driver.maximize_window()
            driver.get(url)
            pagesource = driver.page_source
            soup = BeautifulSoup(pagesource, "html.parser")
            noinputflag, inputbox = inputGet(driver, soup, tag)
            nologoflag, logobox = logoGet(url, driver, soup, tag)                    
            if noinputflag or nologoflag:
                driver.quit()
                continue
            else:               
                if not os.path.exists('D:\\yuxuan\\trainSet\\' + tag):
                    os.makedirs('D:\\yuxuan\\trainSet\\' + tag)  
                    
                imgpath = 'D:\\yuxuan\\trainSet\\' + tag + '\\' + str(tag)+'.png'
                driver.save_screenshot(imgpath)
                # draw bounding box    
                img = cv2.imread('D:\\yuxuan\\trainSet\\'+tag+'\\'+str(tag)+'.png')
                path = 'D:\\yuxuan\\trainSet\\'+tag
                boundingBox(k, path, img, inputbox, logobox)
            driver.quit()
        except TimeoutException:
            driver.quit()            
        except MaxRetryError:
            driver.quit()
        except Exception as e:
            driver.quit()
            print(e)



if __name__ == '__main__': 
    main()