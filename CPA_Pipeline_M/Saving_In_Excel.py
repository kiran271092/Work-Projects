
import fitz
import pandas as pd
import logging
import numpy as np
import time
import os

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger()

class save_headers:

    def open_pdf(self,file_path):
        os.startfile(file_path)
        #time.sleep(5)
        count=0
        while True:
            time.sleep(2)  # wait for 2 seconds
            try:
                os.rename(file_path, file_path)
                if count>0:
                    break
            except:
                if count==0:
                    logger.info("Waiting to close pdf to get values...")
                count = count+1
                continue
        return

    def save_for_m1(self,input_path,temp_Document,claim_list,temp_id):
        
        try:
            path_pdf = os.path.split(input_path)[0]
            name = os.path.split(input_path)[1]
            name = "M1_" + name
            input_path = os.path.join(path_pdf, name)
            self.open_pdf(input_path)
            doc = fitz.open(input_path)
            print(doc.pageCount)
        except Exception as exe_error:
            print("Here")
            logger.info("Unable to open highlighted pdf for M1!!!")
            raise exe_error
        else:
            if len(claim_list)>1:
                start = claim_list[0]
                end = claim_list[1]
            else:
                start = claim_list[0]
                end = int(np.max(temp_Document['Page']))+1
            page_annotation = {}
            page_count = doc.pageCount

            try:
                for ind_page in range(start,end):
                    pts = []
                    page = doc[ind_page-1]
                    annot = page.firstAnnot
                    while annot is not None:
                        pts.append(annot.rect)
                        page_annotation[ind_page] = pts
                        annot = annot.next
                key_list = list(page_annotation.keys())
                word = []
                x1 = []
                y1 = []
                x2 = []
                y2 = []
                for key in key_list:
                    pts = page_annotation[key]
                    print(pts)
                    for pt in pts:
                        pdf_word = list(temp_Document[(temp_Document['Page']==str(key))&(temp_Document['X_start']>=pt[0])&(temp_Document['X_end']<=pt[2])&(temp_Document['Y_avg']>=pt[1])&(temp_Document['Y_avg']<=pt[3])].sort_values(by=['Y_round','X_start'],ascending=True)['Word'].values)
                        if len(pdf_word):
                            pdf_coord = list(pt)
                            pdf_word = ' '.join(pdf_word)
                            word.append(pdf_word)
                            x1.append(pdf_coord[0])
                            y1.append(pdf_coord[1])
                            x2.append(pdf_coord[2])
                            y2.append(pdf_coord[3])
                try:
                    assert len(x1) == len(x2), "Number of coordinates are not equal"
                    assert len(x2) == len(y1), "Number of coordinates are not equal"
                    assert len(y1) == len(y2), "Number of coordinates are not equal"
                    try:
                        try:
                            df = pd.read_csv("Non_table_headers.csv")
                        except:
                            df = pd.read_csv("Non_table_headers.csv",encoding='latin-1')
                        if len(df[df['Id'] == temp_id]) < 1:
                            temp_df = pd.DataFrame()
                            temp_df.loc[:, 'Non_Table_Headers'] = word
                            temp_df.loc[:, 'Start'] = x1
                            temp_df.loc[:, 'End'] = x2
                            temp_df.loc[:, 'Y_start'] = y1
                            temp_df.loc[:, 'Y_end'] = y2
                            temp_df.loc[:, 'Id'] = temp_id
                            df = df.append(temp_df)
                        else:
                            logger.info("Duplicate template id")
                    except:
                        df = pd.DataFrame()
                        df.loc[:, 'Non_Table_Headers'] = word
                        df.loc[:, 'Start'] = x1
                        df.loc[:, 'End'] = x2
                        df.loc[:, 'Y_start'] = y1
                        df.loc[:, 'Y_end'] = y2
                        df.loc[:, 'Id'] = temp_id
                    try:
                        df.to_csv('Non_table_headers.csv', index=False)
                        logger.info("Record entered for non table")
                    except:
                        logger.info("Unable to save headers in excel")
                except AssertionError as asserterror:
                    logger.info(asserterror)
                    raise asserterror
            except Exception as exe_error:
                logger.info("Unable to read highlighted pdf for M1")
        finally:
            try:
                doc.close()
                os.remove(input_path)
            except:
                logger.info("No spacy predictions for M1")

    def save_for_m2(self,input_path,temp_Document,claim_list,temp_id):
        try:
            path_pdf = os.path.split(input_path)[0]
            name = os.path.split(input_path)[1]
            name = "M2_" + name
            input_path = os.path.join(path_pdf, name)
            self.open_pdf(input_path)
            doc = fitz.open(input_path)
        except Exception as exe_error:
            logger.info("Unable to open highlighted pdf for M2!!!")
            raise exe_error
        else:
            if len(claim_list)>1:
                start = claim_list[0]
                end = claim_list[1]
            else:
                start = claim_list[0]
                end = int(np.max(temp_Document['Page']))+1
            page_annotation = {}
            page_count = doc.pageCount
            final_start = []
            final_end = []
            final_y1 = []
            final_y = []
            final_word = []
            pts = []
            try:
                for ind_page in range(start,end):
                    page = doc[ind_page-1]
                    annot = page.firstAnnot
                    while annot is not None:
                        pts.append(annot.rect)
                        page_annotation[ind_page] = pts
                        annot = annot.next
                key_list = list(page_annotation.keys())
                for key in key_list:
                    pts = []
                    pts = page_annotation[key]
                    for pt in pts:
                        temp_doc = temp_Document[(temp_Document['X_start']>=pt[0]) & (temp_Document['Y_avg']>=pt[1]) & (temp_Document['X_end']<=pt[2]) & (temp_Document['Y_avg']<=pt[3]) & (temp_Document['Page']==str(key))].sort_values(by=['X_start','Y_round'])
                        unique_y = temp_doc['Y'].unique()
                        unique_y.sort()
                        word = ""
                        xstart = -1
                        xend = -1
                        y1 = -1
                        y = -1
                        count = 0
                        for ind_y in unique_y:
                            group_doc = temp_doc[temp_doc['Y']==ind_y]
                            if len(group_doc)>0:
                                group_doc = group_doc.sort_values(by=['X_start'])
                                group_words = list(group_doc['Word'].values)
                                group_words = ' '.join(group_words)
                                group_xstart = min(group_doc['X_start'].values)
                                group_xend = max(group_doc['X_end'].values)
                                group_y1 = min(group_doc['Y1'].values)
                                group_y = max(group_doc['Y'].values)
                                if count==0:
                                    word = group_words
                                    xstart = group_xstart
                                    xend = group_xend
                                    y1 = group_y1
                                    y = group_y
                                else :
                                    word = word + " " +group_words
                                    xstart = min(xstart,group_xstart)
                                    xend = max(xend,group_xend)
                                    y1 = min(y1,group_y1)
                                    y = max(y,group_y)
                                count = count+1
                        final_start.append(xstart)
                        final_end.append(xend)
                        final_y1.append(y1)
                        final_y.append(y)
                        final_word.append(word)
                        final_word = [x for _,x in sorted(zip(final_start,final_word))]
                        final_end = [x for _,x in sorted(zip(final_start,final_end))]
                        final_y1 = [x for _,x in sorted(zip(final_start,final_y1))]
                        final_y = [x for _,x in sorted(zip(final_start,final_y))]
                        final_start.sort()
                        count = 0
                        for i in range (0,len(final_word)):
                            if len(final_word[i])<1:
                                count = count+1
                            if len(final_word[i])>=1:
                                break
                        if count>0:
                            for i in range(0,count):
                                final_word.pop(i)
                                final_end.pop(i)
                                final_start.pop(i)
                                final_y1.pop(i)
                                final_y.pop(i)
                            final_word = [x for _,x in sorted(zip(final_start,final_word))]
                            final_end = [x for _,x in sorted(zip(final_start,final_end))]
                            final_y1 = [x for _,x in sorted(zip(final_start,final_y1))]
                            final_y = [x for _,x in sorted(zip(final_start,final_y))]
                            final_start.sort()
                try:
                    assert len(final_start)==len(final_end),"Number of coordinates are not equal"
                    assert len(final_y)==len(final_end),"Number of coordinates are not equal"
                    assert len(final_y) == len(final_y1), "Number of coordinates are not equal"
                    try:
                        df = pd.read_csv("Table_headers.csv")
                        if len(df[df['Id']==temp_id])<1:
                            temp_df = pd.DataFrame()
                            temp_df.loc[:,'Table_Headers'] = final_word
                            temp_df.loc[:,'Start'] = final_start
                            temp_df.loc[:,'End'] = final_end
                            temp_df.loc[:,'Y_start'] = final_y1
                            temp_df.loc[:,'Y_end'] = final_y
                            temp_df.loc[:,'Id'] = temp_id
                            df = df.append(temp_df)
                        else:
                            logger.info("Duplicate template id")
                    except:
                        df = pd.DataFrame()
                        df.loc[:, 'Table_Headers'] = final_word
                        df.loc[:, 'Start'] = final_start
                        df.loc[:, 'End'] = final_end
                        df.loc[:, 'Y_start'] = final_y1
                        df.loc[:, 'Y_end'] = final_y
                        df.loc[:, 'Id'] = temp_id
                    try:
                        df.to_csv('Table_headers.csv',index=False)
                        logger.info("Record entered for table")
                    except:
                        logger.info("Unable to save headers in excel")
                    return final_start,final_end,final_word
                except AssertionError as asserterror:
                    logger.info(asserterror)
                    raise asserterror
            except Exception as exe_error:
                logger.info("Unable to read highlighted pdf for M2")
        finally:
            try:
                doc.close()
                os.remove(input_path)
            except:
                logger.info("No spacy predictions")

