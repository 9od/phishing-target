# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:08:07 2019

@author: syxuan
"""

import os
import re
import cv2
import random
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


def childLogo(link, originurl):
    for child in link.children:
        print(str(child))
        if child is None:
            return link
        elif (('href='+originurl) in str(child) or
               'href="/"' in str(child)):
            return child
    return link


# Get the info of the input box
def inputGet(driver, soup, tag):
    flag = True
    links = driver.find_elements_by_xpath("//input")
    inputs_ = soup.find_all('input')
    inputs=[]
    for inp in inputs_:
        if header(inp):
            continue
        inputs.append(inp)
        
    for k in range(0,len(links)):
#        print('1'+str(inputs[k]))  ###########################
        try:
            size_=links[k].size
        except:
            continue
        if(size_['width']!=0 and size_['height']!=0):
#            print('2'+str(inputs[k]))   ######################
            if (str(inputs[k]).find('search') != -1 or str(inputs[k]).find('Search')!=-1):
                continue
            if (str(inputs[k]).find('user') != -1 or str(inputs[k]).find('email')!=-1 
                 or str(inputs[k]).find('e-mail')!=-1 or str(inputs[k]).find('login')!=-1
                 or str(inputs[k]).find('Email')!=-1 ):
#                print('3'+str(inputs[k]))  ####################
                flag = False
                break
    return flag, links[k].rect

# Get the info of the logo location
def logoGet(url, driver, soup, tag):
    flag = True
    s = findStr(url, '/', 3)
    originurl = url[0:s]+'/'
    
#    links1 = driver.find_element_by_xpath("//div[contains(@id,'logo' or @id,'Logo' or @class,'logo' or @class,'Logo')]")
    links2 = driver.find_elements_by_xpath("//img")   
    links3 = driver.find_elements_by_xpath("//*[name()='svg']")   
    links4 = driver.find_elements_by_xpath("//*[contains(@id,'logo') or contains(@id,'Logo')]")
    links5 = driver.find_elements_by_xpath("//*[contains(@class,'logo') or contains(@class,'Logo')]")        
    links6_ = driver.find_elements_by_xpath("//a[@href]")
    links7 = driver.find_elements_by_xpath("//a")  
    links6=[]
    for link in links6_:
        if link.get_attribute("href")==originurl:
            links6.append(link)   
    links =links2 + links3 + links4 + links5 + links6 + links7
    
#    logos1 = soup.find_all(name='div',attrs={'class':re.compile('logo',flags=re.IGNORECASE),\
#                                             'id':re.compile('logo',flags=re.IGNORECASE)})
    logos2 = soup.find_all(name='img')
    logos3 = soup.find_all(name='svg')
    logos4 = soup.find_all(id=re.compile('logo',flags=re.IGNORECASE))
    logos5 = soup.find_all(class_=re.compile('logo',flags=re.IGNORECASE))
    logos6=[]
    logos7 = soup.find_all(name='a')
    for logo_ in soup.find_all('a', attrs={'href':True }):
        if logo_['href']=='/' or logo_['href']==originurl:
            logos6.append(logo_)   
    logos=[]             
    logos_ = logos2 + logos3 + logos4 + logos5 + logos6 + logos7
    for logo in logos_:
        if (re.compile('logout',flags=re.IGNORECASE).findall(str(logo))!=[]
            or header(logo)):
            continue
        logos.append(logo)
    
    for k in range(0,len(links)):
        print('1'+str(logos[k]))  ###############################
        try:
            size_=links[k].size
        except:
            continue
        if(size_['width']!=0 and size_['height']!=0):
            print('2'+str(logos[k]))  ############################
            if (re.compile('logo',flags=re.IGNORECASE).findall(str(logos[k])) != [] or
                ('href='+originurl) in str(logos[k]) or
                'href="/"' in str(logos[k])) :  
                print('3'+str(logos[k]))  ########################
#                rec=links[k].rect
#                lo=links[k].location
#                wh=links[k].size
#                print(rec['x'], rec['y'], rec['width'], rec['height'])
#                print(lo['x'], lo['y'])
#                print(wh['width'], wh['height'])
                flag = False
                break   
    return flag, links[k].rect

# Draw the bounding box
def boundingBox(k, path, img, inputbox, logobox): 
    # Input
    inputbox['x'] = abs(inputbox['x'])
    inputbox['y'] = abs(inputbox['y'])
    inputpx1 = (int(inputbox['x']-10), int(inputbox['y']-10))
    inputpx2 = (int(inputbox['x']+inputbox['width']+10), int(inputbox['y']+inputbox['height']+10))
    cv2.rectangle(img, inputpx1, inputpx2, (0,0,255), 2)
    cv2.putText(img, 'input', inputpx1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)   
    inputpath = path + '\\input_mark.txt'
    f_new = open(inputpath, 'w')
    f_new.write(str(inputbox))
    f_new.close()
    
    # Logo
    logobox['x'] = abs(logobox['x'])
    logobox['y'] = abs(logobox['y'])
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
    cv2.imwrite('D:\\yuxuan\\trainAccur\\111\\'+str(k)+'.png', img)

# Main function
def main():     
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400',
        'referer':'http://www.taobao.com' }
    cap = DesiredCapabilities.PHANTOMJS.copy() 
    for key, value in headers.items():
        cap['phantomjs.page.customHeaders.{}'.format(key)] = value
    
    urls, targets = urlsGet()   
    random.shuffle(urls)
    random.shuffle(urls)
    urls_ = random.sample(urls, 300)
    
    for k in range(0, len(urls_)):
        try:
            if k==1:
                break
            print(str(k))
            url=urls_[k]          
            tag=targets[k]
            
            ###################################
            url = 'https://sproboticworks.com/login' 
            tag = '111'  #######################

            driver = webdriver.PhantomJS()
            driver.implicitly_wait(10)
            driver.set_page_load_timeout(10)
            driver.set_script_timeout(10)
            driver.maximize_window()
            driver.execute_script("document.body.style.zoom='120 %'")
            driver.get(url)

            pagesource = driver.page_source
            soup = BeautifulSoup(pagesource, "html.parser")
            noinputflag, inputbox = inputGet(driver, soup, tag)
            nologoflag, logobox = logoGet(url, driver, soup, tag)                    
            if noinputflag or nologoflag:
                driver.quit()
                continue
            else:               
                if not os.path.exists('D:\\yuxuan\\trainAccur\\' + tag):
                    os.makedirs('D:\\yuxuan\\trainAccur\\' + tag)  
                    
                imgpath = 'D:\\yuxuan\\trainAccur\\' + tag + '\\' + str(tag)+'.png'
                driver.save_screenshot(imgpath)
                # draw bounding box    
                img = cv2.imread('D:\\yuxuan\\trainAccur\\'+tag+'\\'+str(tag)+'.png')
                path = 'D:\\yuxuan\\trainAccur\\'+tag
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