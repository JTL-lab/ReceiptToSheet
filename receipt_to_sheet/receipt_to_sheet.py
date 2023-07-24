#!/usr/bin/env python

"""
ReceiptToSheet processes receipts saved in PDF format in order to extract data such as the date of purchase, vendor,
items purchased, and total purchase amount and updates a personal finance spreadsheet with this data.

TO BE ADDED:
- Autoencoder trained on PDF and receipts data to preprocess images effectively.
- Classifier trained on past spreadsheet data in order to accurately identify the category of each purchase.
"""

import os
import re
import sys
import tempfile
import argparse
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pytesseract
import cv2


def parse_args(args=None):
    """
    Defines available command line arguments.
    """
    Description = "Processes a directory of PDF receipts and updates a personal finance spreadsheet with their data!"
    Epilog = "Example usage: python receipt_to_sheet.py <PDF_PATH> <SPREADSHEET_PATH>"
    parser = argparse.ArgumentParser(description=Description, epilog=Epilog)
    parser.add_argument('PDF_PATH', metavar='pdf_path', type=str,
                        help='Path to directory containing PDFs.')
    parser.add_argument('SPREADSHEET_PATH', metavar='sheet_path', type=str,
                        help='Path to spreadsheet to update data rows to OR path new spreadsheet should be created in.')
    return parser.parse_args(args)


def preprocess_receipt_image(image):
    """
    Processes PIL images to be in greyscale and sharpened.
    TBA: update this with autoencoder model to preprocess images better!
    """
    # Get image's numerical representation
    image_arr = np.array(image)

    # Convert data values to greyscale equivalent
    greyscale_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)

    # Apply denoising to deal with faded edges, contrast issues
    denoised_arr = cv2.fastNlMeansDenoising(greyscale_arr, None, 15, 15, 20)

    # Apply median filtering for further noise reduction
    filtered_arr = cv2.medianBlur(denoised_arr, 5)  # Adjust kernel size as needed

    # Enhance contrast using Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_arr = clahe.apply(filtered_arr)

    # Convert back to PIL image
    final_image = Image.fromarray(equalized_arr)

    return final_image


def convert_pdf_to_greyscale_image(pdf_path):
    """
    Converts PDF of receipt into a JPEG image using pdf2image library.
    """
    # Obtain PIL image for every page in PDF
    images = convert_from_path(pdf_path)
    greyscale_images = []
    for i, image in enumerate(images):
        preprocessed_image = preprocess_receipt_image(image)
        preprocessed_image.show()
        greyscale_images.append(preprocessed_image)

    return greyscale_images


def extract_image_text(greyscale_image):
    """
    Extracts text from greyscale image of receipt.
    """
    extracted_text = pytesseract.image_to_string(greyscale_image)
    return extracted_text


def process_receipt_text(receipt_text):
    """
    Processes text from receipt image using regular expressions.
    """
    receipt_data_dict = dict()

    # Define patterns for data to extract
    date_pattern = r'\d{2}/\d{2}/\d{4}'  # Matches date format MM/DD/YYYY
    merchant_pattern = r'(?<=\n)[A-Za-z\s\']+'
    # Need to update this: variable success so far :/
    items_pattern = r'(\d+ [A-Za-z\s\']+ \w+ \d+ \$.+\n?)'
    total_price_pattern = r'Total\s*(-?\$[\d.]+)'

    # Extract the date
    date_match = re.search(date_pattern, receipt_text)
    date = date_match.group(0) if date_match else None

    # Extract the merchant
    merchant_match = re.search(merchant_pattern, receipt_text)
    merchant = merchant_match.group(0).strip() if merchant_match else None

    # Extract items purchased
    items = re.findall(items_pattern, receipt_text, re.MULTILINE)

    # Extract the total price
    total_price_match = re.search(total_price_pattern, receipt_text)
    total_price = total_price_match.group(1) if total_price_match else None

    # Printing the extracted information for verification
    #print("Date:", date)
    #print("Merchant:", merchant)
    #print("Items Purchased:", items)
    #print("Total Price:", total_price)

    #print("-------------------------")
    #if date_match:
    #    date = date_match.group(0)
    #    print("Date:", date)
    #if items:
    #    print("Items Purchased:", items)
    #if merchant:
    #    print("Merchant:", merchant)
    #if total_price:
    #    total = float(total_price)
    #    print("Total Price:", total_price)
    #print("-------------------------")

    return receipt_data_dict


def process_pdfs(pdf_dir):
    """
    Returns dictionary where keys correspond to a given PDF name and values correspond to a nested dictionary
    of the content from each PDF page as a string (keys -> page, values -> str content).
    """
    pdf_content_dict = dict()
    pdf_text_data_dict = dict()
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            # Get grayscale image of PDF
            filepath = os.path.join(pdf_dir, filename)
            greyscale_images = convert_pdf_to_greyscale_image(filepath)
            print("RECEIPT: {}".format(filename))
            # Extract text from image
            image_text_content = ''
            for image in greyscale_images:
                pdf_text = extract_image_text(image)
                image_text_content += pdf_text

            # Process text data
            receipt_data = process_receipt_text(image_text_content)

            # Retain data for each file
            pdf_content_dict[filename] = image_text_content
            pdf_text_data_dict[filename] = receipt_data

    return pdf_content_dict, pdf_text_data_dict


def add_data_to_sheet(extracted_data_by_pdf):
    """
    Updates data extracted from new receipts to dataframe.
    """
    data_df = pd.DataFrame(columns=["Date", "Category", "Merchant", "Description", "In", "Out"])
    return


def write_data_to_sheet(extracted_data_df):
    """
    Updates spreadsheet with extracted data.
    """
    return


def train_category_classifier(historical_data_df):
    """
    Trains some classifiers to accurately assign the category for each new purchase based on historical data!
    """
    return

def receipt_to_sheet(pdf_path, spreadsheet_path):
    """
    Driver script to process receipt PDFs into organized purchase data and update spreadsheet accordingly.
    """
    # Process all data
    text_by_pdf, extracted_data_by_pdf = process_pdfs(pdf_path)

    # Update spreadsheet
    add_data_to_sheet(extracted_data_by_pdf)


def main(args=None):
    args = parse_args(args)
    receipt_to_sheet(args.PDF_PATH, args.SPREADSHEET_PATH)


if __name__ == '__main__':
    sys.exit(main())