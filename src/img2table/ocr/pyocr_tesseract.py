# coding: utf-8

import os
import sys
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from typing import List, Iterator, Optional
from PIL import Image as PIL_Image
import codecs
from dataclasses import dataclass

import cv2
import numpy as np
import polars as pl
from bs4 import BeautifulSoup

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe
import pyocr
 

class Pyocr_TesseractOCR(OCRInstance):
    """
    Tesseract-OCR instance
    """
    def __init__(self, image_path, n_threads: int = 1, lang: str = 'eng', psm: int = 11, tessdata_dir: Optional[str] = None):
        """
        Initialization of Tesseract OCR instance
        :param image_path: path of image 
        :param n_threads: number of concurrent threads used for Tesseract
        :param lang: lang parameter used in Tesseract
        :param psm: PSM parameter used in Tesseract
        :param tessdata_dir: directory containing Tesseract traineddata files
        """
        self.image_path = image_path
        
        if isinstance(n_threads, int):
            self.n_threads = n_threads
        else:
            raise TypeError(f"Invalid type {type(n_threads)} for n_threads argument")

        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        if isinstance(psm, int):
            self.psm = psm
        else:
            raise TypeError(f"Invalid type {type(psm)} for psm argument")

        # Create custom environment
        env = os.environ.copy()
        if tessdata_dir:
            env["TESSDATA_PREFIX"] = tessdata_dir
        self.env = env

        # Check if Tesseract is available
        cmd_tess = subprocess.run("tesseract --version", env=self.env, shell=True)
        if cmd_tess.returncode != 0:
            raise EnvironmentError("Tesseract not found in environment. Check variables and PATH")

        # Check if requested languages are available
        try:
            lang_tess = subprocess.check_output("tesseract --list-langs", env=self.env, shell=True).decode()
            for lang in self.lang.split('+'):
                if not any([re.search(fr"\b{lang}\b", line) is not None for line in lang_tess.splitlines()]):
                    raise EnvironmentError(f"Tesseract '{lang}' trainned data cannot be located")
        except subprocess.CalledProcessError:
            raise EnvironmentError("Tesseract trainned data cannot be located.")

    def hocr(self, image: np.ndarray) -> str:
        """
        Get hOCR HTML of an image using Tesseract
        :param image: numpy array representing the image
        :return: hOCR HTML string
        """
        with NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_f:
            tmp_file = tmp_f.name
            # Write image to temporary file
            cv2.imwrite(tmp_file, image)

            # Get hOCR
            hocr = subprocess.check_output(f"tesseract {tmp_file} stdout --psm {self.psm} -l {self.lang} hocr",
                                           env=self.env,
                                           stderr=subprocess.STDOUT,
                                           shell=True)
            
            
            
            #print(type(hocr))
            #print(hocr.decode('utf-8'))
            
        # Remove temporary file
        while os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except PermissionError:
                pass

        return hocr.decode('utf-8')

    def content(self, document: Document) -> Iterator[str]:
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            hocrs = pool.map(self.hocr, document.images)

        return hocrs

    def to_ocr_dataframe(self, content: List[str]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of dataframes for each page
        list_dfs = list()
        
        for page, hocr in enumerate(content):
            # Instantiate HTML parser
            soup = BeautifulSoup(hocr, features='html.parser')

            # Parse all HTML elements
            list_elements = list()
            for element in soup.find_all(class_=True):
                if "word" in element["id"]:
                    continue
                else:
                    # Get element properties
                    d_el = {
                        "page": page,
                        "class": element["class"][0],
                        "id": element["id"],
                        "parent": element.parent.get('id'),
                        "value": re.sub(r"^(\s|\||L|_|;|\*)*$", '', element.string).strip() or None if element.string else None
                    }

                    # Get word confidence
                    str_conf = re.findall(r"x_wconf \d{1,2}", element["title"])
                    if str_conf:
                        d_el["confidence"] = int(str_conf[0].split()[1])
                    else:
                        d_el["confidence"] = None

                    # Get bbox
                    bbox = re.findall(r"bbox \d{1,4} \d{1,4} \d{1,4} \d{1,4}", element["title"])[0]
                    d_el["x1"], d_el["y1"], d_el["x2"], d_el["y2"] = tuple(
                        int(element) for element in re.sub(r"^bbox\s", "", bbox).split())

                    #print(d_el)
                    list_elements.append(d_el)

            
            html_path = self.get_words_hocr()
            with open(html_path) as hocr_words:
                soup = BeautifulSoup(hocr_words, features='html.parser')
                
                
                
                for idx, element in enumerate(soup.find_all(class_=True)):
                    d_el = {
                            "page": 0,
                            "class": element["class"][0],
                            "id": f"word_1_{idx}",
                            "parent": None,
                            "value": re.sub(r"^(\s|\||L|_|;|\*)*$", '', element.string).strip() or None if element.string else None
                        }

                    # Get word confidence
                    str_conf = re.findall(r"x_wconf \d{1,2}", element["title"])
                    if str_conf:
                        d_el["confidence"] = int(str_conf[0].split()[1])
                    else:
                        d_el["confidence"] = None

                    # Get bbox
                    bbox = re.findall(r"bbox \d{1,4} \d{1,4} \d{1,4} \d{1,4}", element["title"])[0]
                    d_el["x1"], d_el["y1"], d_el["x2"], d_el["y2"] = tuple(
                        int(element) for element in re.sub(r"^bbox\s", "", bbox).split())
                    
                    list_elements.append(d_el)
        
        
            # Create dataframe
            if list_elements:
                list_dfs.append(pl.LazyFrame(data=list_elements, schema=self.pl_schema))
              
            print(html_path)
            while os.path.exists(html_path):
                try:
                    os.remove(html_path)
                    print("delete")
                except PermissionError:
                    pass  

        return OCRDataframe(df=pl.concat(list_dfs)) if list_dfs else None
    
    def get_words_hocr(self, tesseract_layout=4):
        out = subprocess.check_output("where tesseract",stderr=subprocess.STDOUT,shell=True)
        path = out.decode("utf-8")
        path = re.sub("\r\n","",path)
        
        pyocr.tesseract.TESSERACT_CMD = path


        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise ModuleNotFoundError("No OCR tool found")

        # The tools are returned in the recommended order of usage
        tool = tools[0]
        builder = pyocr.builders.WordBoxBuilder(tesseract_layout=tesseract_layout)
        
        image = PIL_Image.open(self.image_path)
        words = tool.image_to_string(
            image,
            lang=self.lang,
            builder=builder
        )
        with NamedTemporaryFile(suffix='.html', delete=False) as tmp_f:
            tmp_file = tmp_f.name
            with codecs.open(tmp_file, 'w', encoding='utf-8') as file_descriptor:
                builder.write_file(file_descriptor,words)
        
        return tmp_file
        
        
