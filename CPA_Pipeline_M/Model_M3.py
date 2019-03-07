
import pandas as pd
import re
import math
import numpy as np
import string
from sklearn.neighbors import NearestNeighbors

class get_output:

    def getPositionOfSeperator(self,word):
        if (":" in word):
            return word.find(":")
        return word.find(",")

    def isSpecialChar(self,word):
        invalidChars = set(string.punctuation.replace(":", ""))
        if any(char in invalidChars for char in word):
            return True
        else:
            return False

    def create_labels_list(self,labels):
        labels.sort(key=lambda x: len(x.split()), reverse=True)
        return labels

    def ContainsDate(self,X, arguments=None):
        regEx = "Jan[\s+|\,|\'|\-|\/|\d]|Feb[\s+|\,|\'|\-|\/|\d]|Mar[\s+|\,|\'|\-|\/|\d]|Apr[\s+|\,|\'|\-|\/|\d]|May[\s+|\,|\'|\-|\/|\d]|Jun[\s+|\,|\'|\-|\/|\d]|June[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|July[\s+|\,|\'|\-|\/|\d]|Aug[\s+|\,|\'|\-|\/|\d]|Sep[\s+|\,|\'|\-|\/|\d]|Oct[\s+|\,|\'|\-|\/|\d]|Nov[\s+|\,|\'|\-|\/|\d]|Dec[\s+|\,|\'|\-|\/|\d]|December|January|February|March|April|August|September|October|November|December?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|\d{1,2}[\-|\,|\/|\.]{1}\d{1,2}[\-|\,|\/|\.]{1}\d{2,4}"
        pattern = re.compile(regEx, re.IGNORECASE)
        X1 = []
        X1 = X
        y = []
        for i in range(len(X1)):
            if (re.search(pattern, X1[i]) is not None):
                y.append(True)
            else:
                y.append(False)

        for i in range(len(X1)):
            s = X1[i]
            digits = re.findall(r"[0-9]{6,8}", s, flags=re.MULTILINE)
            for j in digits:
                if (len(j) == 6):
                    mm = (ord(j[0]) - 48) * 10 + (ord(j[1]) - 48)
                    dd = (ord(j[2]) - 48) * 10 + (ord(j[3]) - 48)
                    yy = (ord(j[4]) - 48) * 10 + (ord(j[5]) - 48)
                    if (((mm > 0) and (mm <= 12) and (dd > 0) and (dd <= 31) and (yy >= 16) and (yy <= 18)) or (
                            (mm > 0) and (mm <= 31) and (dd > 0) and (dd <= 12) and (yy >= 16) and (yy <= 18))):
                        y[i] = 1
                        break

                if (len(j) == 8):

                    mm = (ord(j[0]) - 48) * 10 + (ord(j[1]) - 48)
                    dd = (ord(j[2]) - 48) * 10 + (ord(j[3]) - 48)
                    yyyy = (ord(j[4]) - 48) * 1000 + (ord(j[5]) - 48) * 100 + (ord(j[6]) - 48) * 10 + (ord(j[7]) - 48)
                    if (((mm > 0) and (mm <= 12) and (dd > 0) and (dd <= 31) and (yyyy >= 2016) and (yyyy <= 2018)) or (
                            (mm > 0) and (mm <= 31) and (dd > 0) and (dd <= 12) and (yyyy >= 2016) and (yyyy <= 2018))):
                        y[i] = 1
                        break
        y = np.array(y)
        return y

    def is_pattern(self,pattern, x):
        if (re.search(pattern, x) is not None):
            return True
        else:
            return False

    def has_tag(self,label):
        small_label = label.lower()
        tags = ["#", "id", " no", "number", "date", "amount"]
        if (any((word in small_label) for word in tags)):
            return True
        return False

    def tagging(self,all_neighbours, label):
        small_label = label.lower()
        if (any(word in small_label for word in ["#", "id", " no", "number"])):
            return all_neighbours[(all_neighbours['is_static'] == 1) | (
                        (all_neighbours['is_static'] == 0) & (all_neighbours['id_tag'] == True))]
        elif ("date" in small_label):
            return all_neighbours[(all_neighbours['is_static'] == 1) | (
                        (all_neighbours['is_static'] == 0) & (all_neighbours['date_tag'] == True))]
        elif ("amount" in small_label):
            return all_neighbours[(all_neighbours['is_static'] == 1) | (
                        (all_neighbours['is_static'] == 0) & (all_neighbours['amount_tag'] == True))]
        else:
            return all_neighbours

    def create_claims(self,claim_no_list,last_page_number):
        complete_list=claim_no_list.copy()
        complete_list.append(last_page_number)
        final_claims=[]
        for i in range(len(complete_list)-1):
            if i==(len(complete_list)-2):
                a=(complete_list[i],complete_list[i+1])
                final_claims.append(a)
            else:
                a=(complete_list[i],complete_list[i+1]-1)
                final_claims.append(a)
        return final_claims

    def create_all_Tags(self,Document):
        #for dates or of 1st and 2nd
        pattern_yy_mm_dd_beginner=re.compile("^((20)\d\d[- /.])(((1[012]|0?[1-9])[-/.])|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/.])(3[01]|[12][0-9]|0?[1-9][- /.$]?)(?=\s|\\.|$)",re.IGNORECASE)
        pattern_yy_mm_dd=re.compile("^((20)\d\d[- /.])(((1[012]|0?[1-9])[-/.])|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/. ])(3[01]|[12][0-9]|0?[1-9][- /.$]?)(|.|$)",re.IGNORECASE)
        pattern_mm_dd_yy_beginner=re.compile("^(((1[012]|0?[1-9])[-/.])|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/. ])((3[01]|[12][0-9]|0?[1-9])[-/.$])((19|20)?\d{2})(?=.|$|\s|\d)",re.IGNORECASE)
        pattern_mm_dd_yy=re.compile("^(((1[012]|0?[1-9])[-/.])|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/. ])((3[01]|[12][0-9]|0?[1-9])[-,/.$ ]{0,2})((19|20)?\d{2})(?=.|$|\s|\d)",re.IGNORECASE)
        Document['date_tag']=Document['Word'].apply(lambda x: ((self.is_pattern(pattern_yy_mm_dd_beginner,x)|self.is_pattern(pattern_yy_mm_dd,x)|self.is_pattern(pattern_mm_dd_yy_beginner,x)|self.is_pattern(pattern_mm_dd_yy,x))|(self.ContainsDate([x])[0])))
        regex_amount='[$][\d\.\,][\.\$0-9]*$'
        regex_integer='[+-]?[0-9][0-9]*'
        regex_float='[+-]?[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?'
        regex_not='["/(){}!#a-zA-Z-]'
        regex_amount_pattern = re.compile(regex_amount,re.IGNORECASE)
        regex_integer_pattern = re.compile(regex_integer,re.IGNORECASE)
        regex_float_pattern = re.compile(regex_float,re.IGNORECASE)
        regex_not_pattern = re.compile(regex_not)
        Document['amount_tag']= Document['Word'].apply(lambda x: ((self.is_pattern(regex_amount_pattern,x)|self.is_pattern(regex_integer_pattern,x)|self.is_pattern(regex_float_pattern,x))&(not self.is_pattern(regex_not_pattern,x))))
        regex_id1='([A-Z0-9]*[0-9]+[A-Z0-9]*)'
        regex_id2='(^(?![\-]))[\-0-9]+([0-9]+$)'
        regex_id3='[,."/$(){}!a-z]'
        pattern1 = re.compile(regex_id1,re.IGNORECASE)
        pattern2 = re.compile(regex_id2,re.IGNORECASE)
        pattern3 = re.compile(regex_id3)
        Document['id_tag']=Document['Word'].apply(lambda x: ((self.is_pattern(pattern1,x)|self.is_pattern(pattern2,x))&(not self.is_pattern(pattern3,x))))
        Document['amount_tag']=np.where(Document['date_tag']==True,False,Document['amount_tag'])
        return Document

    def remove_some_characters(self,Document):
        return Document[~(Document['Word']==":")]

    def change_document_indexes(self,Transformed_Document):
        Transformed_Document.reset_index(inplace=True,drop=True)
        Transformed_Document['word_id']=range(0,(len(Transformed_Document)))
        return Transformed_Document

    def Merge_line_wise(self,T_Document):
        Merged_Document=pd.DataFrame()
        row_temp=pd.DataFrame()
        for index,row_m in T_Document.iterrows():
            if re.match('^[|_]+$', row_m['Word']) is not None:
                Merged_Document=pd.concat([Merged_Document,pd.DataFrame(row_temp).T],axis=0)
                continue
            else:
                if (len(row_temp)==0):
                    row_temp=row_m
                else:
                    if (row_m['X_start'] < (row_temp['X_end'] + 2.9*row_temp['Char_Width'])) & (row_m['is_label'] == 0) & (row_temp['date_tag']== row_m['date_tag']) &(row_m['is_label'] == row_temp['is_label'])&(row_m['is_header_label'] == row_temp['is_header_label'])&(row_temp['is_header_label']==0)&(row_m['Y'] == row_temp['Y'])& (row_m['Page'] == row_temp['Page']) & (row_temp['Word'][-1] != ':'):
                        row_temp['Word']=(row_temp["Word"]+" "+row_m['Word'])
                        row_temp['X_start'] = row_temp['X_start']
                        row_temp['X_end'] = row_m['X_end']
                        row_temp['Y'] = row_m['Y']
                        row_temp['Page'] = row_m['Page']
                        row_temp['X_center'] = (row_temp['X_start'] + row_m['X_end'])/2
                        row_temp['Char_Width'] = (row_temp['Char_Width'] + row_m['Char_Width'])/2
                        row_temp['Length'] = (row_temp['Length']+row_m['Length']+1)
                        row_temp['date_tag'] = (row_temp['date_tag']|row_m['date_tag'])
                        row_temp['amount_tag'] = (row_temp['amount_tag']|row_m['amount_tag'])
                        row_temp['id_tag'] = (row_temp['id_tag']|row_m['id_tag'])
                    else:
                        Merged_Document=pd.concat([Merged_Document,pd.DataFrame(row_temp).T],axis=0)
                        row_temp=row_m
        Merged_Document=pd.concat([Merged_Document,pd.DataFrame(row_temp).T],axis=0)
        return Merged_Document

    def transform_doc_with_tab(self,Document,table_labels_dictionary):#Page Level only
        Document['is_header_label']=0
        for key in table_labels_dictionary.keys():
            co_ordinates=table_labels_dictionary[key]
            all_of_label=[]
            for label_part in key.split():
                can_be_label=Document.loc[((abs(len(label_part)-Document['Length'])<=2)&(Document['Word'].str.contains(label_part))),:].reset_index(drop=True)
                if (can_be_label.shape[0])>0:
                    can_be_label['euclidean_distance']=can_be_label.apply(lambda x: math.sqrt(((x['X_start']-co_ordinates[0])**2)+((x['Y1']-co_ordinates[1])**2)+((x['X_end']-co_ordinates[2])**2)+((x['Y2']-co_ordinates[3])**2)),axis=1)
                    can_be_label.sort_values(by='euclidean_distance',ascending=True,inplace=True)
                    all_of_label.append(can_be_label['word_id'].head(1).values[0])
                else:
                    break#no occurance of this part
            if len(all_of_label)==1:
                Document.loc[(Document['word_id']==all_of_label[0]),'Word']=key
                Document.loc[(Document['word_id']==all_of_label[0]),'is_header_label']=1
            elif len(all_of_label)>1:
                Document.loc[(Document['word_id']==all_of_label[0]),'Word']=key
                Document.loc[(Document['word_id']==all_of_label[0]),'is_header_label']=1
                Document.loc[(Document['word_id']==all_of_label[0]),'X_start']=Document.loc[(Document['word_id'].isin(all_of_label)),'X_start'].min()
                Document.loc[(Document['word_id']==all_of_label[0]),'Y1'] = Document.loc[(Document['word_id'].isin(all_of_label)),'Y1'].unique().min()
                Document.loc[(Document['word_id']==all_of_label[0]),'Y2'] = Document.loc[(Document['word_id'].isin(all_of_label)),'Y2'].unique().max()
                Document.loc[(Document['word_id']==all_of_label[0]),'Y'] = Document.loc[(Document['word_id']==all_of_label[0]),'Y2']
                Document.loc[(Document['word_id']==all_of_label[0]),'X_end']= Document.loc[(Document['word_id'].isin(all_of_label)),'X_end'].max()
                Document.loc[(Document['word_id']==all_of_label[0]),'X_center']=(Document.loc[(Document['word_id']==all_of_label[0]),'X_start'].values[0]+Document.loc[(Document['word_id']==all_of_label[0]),'X_end'].values[0])/2
                Document=Document[~(Document['word_id'].isin(all_of_label[1:]))]
            else:
                continue
        return Document

    def transform_key_value_label_doc(self,Document,static_label_dictionary,labels_left):#also page level only
        Document['is_label']=0
        for key in labels_left:
            co_ordinates=static_label_dictionary[key]
            all_of_label=[]
            for label_part in key.split():
                can_be_label=Document.loc[((abs(len(label_part)-Document['Length'])<=2)&(Document['Word'].str.contains(label_part))),:].reset_index(drop=True)
                if (can_be_label.shape[0])>0:
                    can_be_label['euclidean_distance']=can_be_label.apply(lambda x: math.sqrt(((x['X_start']-co_ordinates[0])**2)+((x['Y1']-co_ordinates[1])**2)+((x['X_end']-co_ordinates[2])**2)+((x['Y2']-co_ordinates[3])**2)),axis=1)
                    can_be_label.sort_values(by='euclidean_distance',ascending=True,inplace=True)
                    all_of_label.append(can_be_label['word_id'].head(1).values[0])
                else:
                    break#no occurance of this part
            if len(all_of_label)==1:
                Document.loc[(Document['word_id']==all_of_label[0]),'Word']=key
                Document.loc[(Document['word_id']==all_of_label[0]),'is_label']=1
            elif len(all_of_label)>1:
                Document.loc[(Document['word_id']==all_of_label[0]),'Word']=key
                Document.loc[(Document['word_id']==all_of_label[0]),'is_label']=1
                Document.loc[(Document['word_id']==all_of_label[0]),'X_start']=Document.loc[(Document['word_id'].isin(all_of_label)),'X_start'].min()
                Document.loc[(Document['word_id']==all_of_label[0]),'Y1'] = Document.loc[(Document['word_id'].isin(all_of_label)),'Y1'].unique().min()
                Document.loc[(Document['word_id']==all_of_label[0]),'Y2'] = Document.loc[(Document['word_id'].isin(all_of_label)),'Y2'].unique().max()
                Document.loc[(Document['word_id']==all_of_label[0]),'Y'] = Document.loc[(Document['word_id']==all_of_label[0]),'Y2']
                Document.loc[(Document['word_id']==all_of_label[0]),'X_end']= Document.loc[(Document['word_id'].isin(all_of_label)),'X_end'].max()
                Document.loc[(Document['word_id']==all_of_label[0]),'X_center']=(Document.loc[(Document['word_id']==all_of_label[0]),'X_start'].values[0]+Document.loc[(Document['word_id']==all_of_label[0]),'X_end'].values[0])/2
                Document=Document[~(Document['word_id'].isin(all_of_label[1:]))]
            else:
                #length of list is 0
                continue
        return Document

    def makePage(self,Page,Non_table_labels_dictionary,Table_header_labels,labels_left):
        Page=self.create_all_Tags(Page)
        Page=self.transform_doc_with_tab(Page,Table_header_labels)
        Page=self.change_document_indexes(Page)
        Page=self.remove_some_characters(Page)
        Transformed_Page=self.transform_key_value_label_doc(Page,Non_table_labels_dictionary,labels_left)
        Transformed_Page=self.change_document_indexes(Transformed_Page)
        Transformed_Page=self.Merge_line_wise(Transformed_Page)
        Transformed_Page=self.change_document_indexes(Transformed_Page)
        return Transformed_Page

    def nearest_neighbours_on_Page(self,Transformed_Page_with_Labels,no_of_proximities):
        page_nn_object=NearestNeighbors(n_neighbors=no_of_proximities, metric="euclidean")
        page_nn_object.fit(Transformed_Page_with_Labels[['X_start','Y']])
        return page_nn_object

    def make_labels_on_Page(self,Transformed_Page,labels_left):
        labels_left_update=labels_left.copy()
        default_statics=['page','Page']
        Transformed_Page['is_static']=0
        Transformed_Page['is_static']=np.where((Transformed_Page['is_header_label']==1)|(Transformed_Page['is_label']==1),1,0)
        Transformed_Page['is_static']=Transformed_Page.apply(lambda x: 1 if ((any(word in x['Word'] for word in default_statics))|(x['is_static']==1)) else 0,axis=1)
        labels_found=Transformed_Page[Transformed_Page['is_label']==1]['Word'].values.tolist()
        for lab in labels_found:
            labels_left_update.remove(lab)
        return Transformed_Page,labels_left_update

    def get_claims_values(self,Transformed_Page_with_Labels,Page_NN_Object,Non_table_labels_dictionary,no_of_proximities,labels_left,distance_threshold,inclination_threshold):
        results_dictionary=self.cpa_algorithm(Transformed_Page_with_Labels,Page_NN_Object,Non_table_labels_dictionary,no_of_proximities,labels_left)
        Page_Answers=self.postProcessing(results_dictionary,labels_left,distance_threshold,inclination_threshold)
        return Page_Answers

    def get_one_cordinates(self,word_corpus,word):
        return word_corpus.loc[(word_corpus['Word']==word)&(word_corpus['is_label']==1),:]

    def get_x_y_cordinates_id(self,word_corpus,ids):
        return word_corpus.loc[word_corpus['word_id']==ids,:]

    def get_value_using_recursive_nearest_neighbours_per_Page(self,row,Page_Corpus,Page_NN_Object,label,no_of_neighbours):
        Neighbours_id=[]
        Neighbours_weights=[]
        for lda in ['X_start','X_center','X_end']:
            Neigh_id=Page_NN_Object.kneighbors(pd.DataFrame(row).T[[lda,'Y']], n_neighbors=no_of_neighbours, return_distance=True)[1][0]
            Neigh_weights=Page_NN_Object.kneighbors(pd.DataFrame(row).T[[lda,'Y']], n_neighbors=no_of_neighbours, return_distance=True)[0][0]
            Neighbours_id=Neighbours_id+Neigh_id.tolist()
            Neighbours_weights=Neighbours_weights+Neigh_weights.tolist()
        x_val=pd.DataFrame(row).T['X_center'].values[0]
        x_start_val=pd.DataFrame(row).T['X_start'].values[0]
        y_val=pd.DataFrame(row).T['Y'].values[0]
        nearest_words_static=Page_Corpus.loc[Neighbours_id,'is_static'].values
        all_word_co_ordinates=pd.DataFrame()
        for ids in Neighbours_id:
            word_co_ordinate=self.get_x_y_cordinates_id(Page_Corpus,ids)
            all_word_co_ordinates=pd.concat([all_word_co_ordinates,word_co_ordinate],axis=0)
        all_word_co_ordinates.reset_index(inplace=True,drop=True)
        all_neighbours=pd.DataFrame()
        all_neighbours['label_weight']=Neighbours_weights
        all_neighbours=pd.concat([all_neighbours,all_word_co_ordinates],axis=1)
        all_neighbours.sort_values('label_weight',ascending=True,inplace=True)
        cols_list=all_neighbours.columns.tolist()
        cols_list.remove('label_weight')
        all_neighbours.drop_duplicates(subset=cols_list,keep='first',inplace=True)
        all_neighbours=all_neighbours.loc[all_neighbours['Word']!=label,:]
        all_neighbours=all_neighbours[((all_neighbours['Y']==y_val)&(all_neighbours['X_start']>x_start_val))|((all_neighbours['Y']>y_val)&(all_neighbours['X_end']>x_start_val))]
        dynamic_neighbours=all_neighbours[all_neighbours['is_static']==0]
        static_neighbours=all_neighbours[all_neighbours['is_static']==1]
        return static_neighbours,dynamic_neighbours

    def cpa_algorithm(self,Transformed_Page_with_Labels,Page_NN_Object,Non_table_labels_dictionary,no_of_proximities,labels_left):
        results={}
        for current_label in labels_left:
            value=[]
            if len(self.get_one_cordinates(Transformed_Page_with_Labels,current_label))>0:
                for index,row in self.get_one_cordinates(Transformed_Page_with_Labels,current_label).iterrows():
                    statics,dynamics=self.get_value_using_recursive_nearest_neighbours_per_Page(row,Transformed_Page_with_Labels,Page_NN_Object,current_label,no_of_proximities)
                    statics=statics[["Word","label_weight","X_center","Y","X_start","X_end",'word_id']]
                    dynamics['scope']='in'
                    dynamics['removed_by']="Not Removed"
                    for index_d,row_d in dynamics.iterrows():
                        min_distance=row_d['label_weight']
                        min_stat_distance=100000
                        removed_by="Not Removed"
                        for index,row in statics.iterrows():
                            if(((row_d['Y']==row['Y'])&(row_d['X_start']>row['X_start']))|((row_d['Y']>row['Y'])&(row_d['X_end']>row['X_start']))):
                                a_1=np.sqrt(np.square(np.subtract(row_d['X_start'],row['X_start']))+np.square(np.subtract(row_d['Y'],row['Y'])))
                                b_1=np.sqrt(np.square(np.subtract(row_d['X_start'],row['X_center']))+np.square(np.subtract(row_d['Y'],row['Y'])))
                                c_1=np.sqrt(np.square(np.subtract(row_d['X_start'],row['X_end']))+np.square(np.subtract(row_d['Y'],row['Y'])))
                                distance=min(a_1,b_1,c_1)
                                if distance<min_stat_distance:
                                    min_stat_distance=distance
                                    removed_by=row['Word']
                        if (min_distance>min_stat_distance)&(min_distance!=100000):
                            dynamics.ix[index_d,"scope"]='out'
                            dynamics.ix[index_d,"removed_by"]=removed_by
                    In_scope_dynamics=dynamics[dynamics["scope"]=='in']
                    if self.has_tag(current_label):
                        In_scope_dynamics_with_tag=self.tagging(In_scope_dynamics,current_label)
                        if ((In_scope_dynamics_with_tag.shape[0])!=0):
                            final_dynamics=In_scope_dynamics_with_tag
                        else:
                            final_dynamics=In_scope_dynamics
                    else:
                        final_dynamics=In_scope_dynamics
                    value.append(final_dynamics.sort_values(by='label_weight',ascending=True))
            results[current_label]=value
        return results

    def postProcessing(self,results_dictionary,labels_left,distance_threshold,inclination_threshold):
        closest_ids=[]
        for stat_label in labels_left:
            result_list=list(results_dictionary[stat_label])
            no_of_times_that_label_exists=len(result_list)
            for iterate in range(no_of_times_that_label_exists):
                if result_list[iterate].shape[0]!=0:
                    closest_ids.append(result_list[iterate]['word_id'].head(1).values[0])
                else:
                    closest_ids.append(-1)
        Final_result=pd.DataFrame(columns=['Label','Value'])
        for label in labels_left:
            result_list=list(results_dictionary[label])
            no_of_times_that_label_exists=len(result_list)
            for iterate in range(no_of_times_that_label_exists):
                final_dynamics=result_list[iterate]
                if len(final_dynamics)>0:
                    first_value_distance=final_dynamics['label_weight'].head(1).values[0]
                    if first_value_distance>distance_threshold:
                        final_dynamics=pd.DataFrame(columns=final_dynamics.columns.tolist())
                if len(final_dynamics)>0:
                    final_dynamics['Inclination_diff']=final_dynamics['label_weight'].rolling(2).apply(lambda x: x[0] - x[1])
                    final_dynamics['Inclination_diff'].fillna(0.0,inplace=True)
                    final_dynamics['Inclination_diff']=final_dynamics['Inclination_diff'].abs()
                    weight=final_dynamics[final_dynamics.Inclination_diff > inclination_threshold]['label_weight']
                    if len(weight)>0:
                        filter_wt=weight.values[0]
                        final_dynamics=final_dynamics[final_dynamics['label_weight']<filter_wt][['Word','label_weight','X_start','X_center','X_end','word_id','Y']]
                    else:
                        final_dynamics=final_dynamics[['Word','label_weight','X_start','X_center','X_end','word_id','Y']]
                final_dynamics.reset_index(inplace=True,drop=True)
                final_dynamics=final_dynamics[final_dynamics.index==0].append(final_dynamics[1:][~final_dynamics['word_id'][1:].isin(closest_ids)])
                if self.has_tag(label):
                    final_dynamics=final_dynamics.head(1)
                else:
                    final_dynamics=final_dynamics.sort_values(['Y','X_start'],ascending=True)
                Final_result=Final_result.append({'Label':label, 'Value':' '.join(final_dynamics['Word'])}, ignore_index=True)
        return Final_result

    def update_dictionary(self,Non_table_labels_dictionary,Table_header_labels):
        Non_table_labels_dictionary2={}
        Table_header_labels2={}
        for label in Non_table_labels_dictionary.keys():
            #handling empty labels or single sample place
            if (len(label)>0)&(label!=' '):
                temp_list=Non_table_labels_dictionary[label]
                temp_list[1]=(2*temp_list[1])
                temp_list[3]=(2*temp_list[3])
                Non_table_labels_dictionary2[label]=temp_list
        for label in Table_header_labels.keys():
            if (len(label) > 0) & (label != ' '):
                temp_list=Table_header_labels[label]
                temp_list[1]=(2*temp_list[1])
                temp_list[3]=(2*temp_list[3])
                Table_header_labels2[label]=temp_list
        return Non_table_labels_dictionary2,Table_header_labels2


    def cpa_model3(self, path_of_claim_in_pdf_format, Non_table_labels_dictionary, Table_header_labels,
                   claim_no_list=[1], Document = 0, no_of_proximities=15, distance_threshold=100, inclination_threshold=40):
        """ Cpa Algorithm for a pdf

        Parameters
        ----------
        path_of_claim_in_pdf_format : absolute system path of pdf file

        Non_tabel_labels_dictionary : dictionary of non-table labels in the claim
            as dictionary having key as label name and value as list of four
            coordinates as [x1,y1,x2,y2]

        Table_header_labels : dictionary of table labels in the claim
            as dictionary having key as label name and value as list of four
            coordinates as [x1,y1,x2,y2]

        claim_no_list : [c1,c2,c3] list of claim number means the pdf contains
            three claims first start from c1, second from c2, third from c3
            (default:[1]- means by default the document is of a single claim per
            pdf and it starts from page number 1 till the end)

        no_of_proximities : no of nearest neighbours to look for the answer
            (default: 15 - means nearby 15 merged words to look for the answer)

        distance_threshold : minimum distance from the label to have an answer say the label
            is Invoice Number and in the radius of 200 units there is no word, hence we will
            assume there is no answer for that label(default: 100 units)

        inclination_threshold : inclination for the next nearest word to be part of the whole
            answer(e.g if label is Address and nearest value is 301 and next nearest is Baidu
            and the distance b/w both are less than inclincation then both words as whole will
            be considered as the part of the answer)(default: 40 units)

        """
        get_path_broken = path_of_claim_in_pdf_format.split('\\')
        filename = get_path_broken[len(get_path_broken) - 1].split('.')[0]
        directory = path_of_claim_in_pdf_format.rsplit(sep='\\', maxsplit=1)[0]
        # Document = get_Document_Corpus(file_path=path_of_claim_in_pdf_format)
        # Document['Y'] = 2 * np.round(Document['Y'], 2)
        # Document['Y1'] = 2 * np.round(Document['Y1'], 2)
        # Document['Y2'] = 2 * np.round(Document['Y2'], 2)
        # function  will create the document in dataframe form with all pages
        Non_table_labels_dictionary2, Table_header_labels2 = self.update_dictionary(Non_table_labels_dictionary,
                                                                               Table_header_labels)
        no_of_claims = len(claim_no_list)
        last_page_number = max(Document['Page'].astype('int32'))
        pages = [i + 1 for i in range(last_page_number)]
        claims = self.create_claims(claim_no_list, last_page_number)
        # now we need to loop through each claim
        for claim_start, claim_end in claims:
            # make_no_of_labels_full
            labels_left = list(Non_table_labels_dictionary2.keys())
            Claim_Answer = pd.DataFrame()
            for page_no in range(claim_start, claim_end + 1):
                if (len(labels_left) > 0):
                    Page = Document[Document['Page'] == str(page_no)]
                    Transformed_Page = self.makePage(Page, Non_table_labels_dictionary2, Table_header_labels2, labels_left)
                    Transformed_Page_with_Labels, labels_left_update = self.make_labels_on_Page(Transformed_Page, labels_left)
                    Page_NN_Object = self.nearest_neighbours_on_Page(Transformed_Page_with_Labels, no_of_proximities)
                    Page_Answers = self.get_claims_values(Transformed_Page_with_Labels, Page_NN_Object,
                                                     Non_table_labels_dictionary2, no_of_proximities, labels_left,
                                                     distance_threshold, inclination_threshold)
                    Claim_Answer = pd.concat([Claim_Answer, Page_Answers], axis=0)
                else:
                    break
                labels_left = labels_left_update
            Claim_Answer.to_csv(
                directory + '\\' + filename + '_claim_from_' + str(claim_start) + '_to_' + str(claim_end) + '.csv')
        print("All Claims Generated.")
        return

class extract_data:

    def get_labels(self,input_excel_M1,input_excel_M2,temp_id):
        try:
            non_table = pd.read_csv(input_excel_M1)
        except:
            non_table = pd.read_csv(input_excel_M1,encoding='latin-1')
        try:
            table = pd.read_csv(input_excel_M2)
        except:
            table = pd.read_csv(input_excel_M2, encoding='latin-1')
        non_table = non_table[non_table['Id']==temp_id]
        table = table[table['Id']==temp_id]
        coord = []
        t1_non_table_label = {}
        t1_table_label = {}
        for index,cell in non_table.iterrows():
            coord.append(cell['Start'])
            coord.append(cell['Y_start'])
            coord.append(cell['End'])
            coord.append(cell['Y_end'])
            t1_non_table_label[cell['Non_Table_Headers']] = coord
            coord = []
        for index,cell in table.iterrows():
            coord.append(cell['Start'])
            coord.append(cell['Y_start'])
            coord.append(cell['End'])
            coord.append(cell['Y_end'])
            t1_table_label[cell['Table_Headers']] = coord
            coord = []
        return t1_non_table_label,t1_table_label

    def main(self,pdf_name,input_excel_M1,input_excel_M2,claim_list,Document_corpus,temp_id):
        output_object = get_output()
        t1_non_table_label,t1_table_label = self.get_labels(input_excel_M1,input_excel_M2,temp_id)
        print(pdf_name)
        output_object.cpa_model3(path_of_claim_in_pdf_format=pdf_name,Non_table_labels_dictionary=t1_non_table_label,Table_header_labels=t1_table_label,claim_no_list=claim_list,Document=Document_corpus,no_of_proximities=15,distance_threshold=100,inclination_threshold=40)
