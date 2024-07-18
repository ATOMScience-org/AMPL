import urllib.request as urlreq
import io
import json
import pandas as pd

# ******************************************************************************************************************************************
def download_smiles(myList,intv=1) :
    """Retrieve canonical SMILES strings for a list of input INCHIKEYS.
    Will return only one SMILES string per INCHIKEY.  If there are multiple values returned, the first is retained and the others are returned in a the discard_lst.  INCHIKEYS that fail to return a SMILES string are put in the fail_lst

    Args:
        myList (list): List of INCHIKEYS

        intv (1): number of INCHIKEYS to submit queries for in one request, default is 1

    Returns:
        list of SMILES strings corresponding to INCHIKEYS

        list of INCHIKEYS, which failed to return a SMILES string

        list of CIDs and SMILES, which were returned beyond the first CID and SMILE found for input INCHIKEY
    """
    ncmpds=len(myList)
    smiles_lst,cid_lst,inchikey_lst=[],[],[]
    sublst=""
    fail_lst=[]
    discard_lst=[]
    for it in range(0,ncmpds,intv) :
        if (it+intv) > ncmpds :
            upbnd=ncmpds
        else :
            upbnd=it+intv
        sublst=myList[it:upbnd]
        inchikey = ','.join(map(str,sublst)) 
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"+inchikey+"/property/CanonicalSMILES/CSV"
        try :
            response = urlreq.urlopen(url)
            html = response.read()
        except :
            fail_lst.append(inchikey)
            continue
        f=io.BytesIO(html)
        cnt=0
        for l in f :
            l=l.decode("utf-8") 
            l=l.rstrip()
            vals=l.split(',')
            if vals[0] == '"CID"' :
                continue
            if cnt > 0:
                #print("more than one SMILES returned, discarding. Appear to be multiple CID values",vals)
                #print("using",cid_lst[-1],smiles_lst[-1],inchikey_lst[-1])
                discard_lst.append(vals)
                break
            
            cid_lst.append(vals[0])
            sstr=vals[1].replace('"','')
            smiles_lst.append(vals[1])    
            inchikey_lst.append(myList[it+cnt])
            cnt+=1
        if cnt != len(sublst) :
            print("warning, multiple SMILES for this inchikey key",cnt,len(sublst),sublst)
    save_smiles_df=pd.DataFrame( {'CID' : cid_lst, 'standard_inchi_key' :inchikey_lst, 'smiles' : smiles_lst})
    return save_smiles_df,fail_lst,discard_lst


#******************************************************************************************************************************************
def download_bioactivity_assay(myList,intv=1) :
    """Retrieve summary info on bioactivity assays.

    Args:
        myList (list): List of PubChem AIDs (bioactivity assay ids)

        intv (1): number of INCHIKEYS to submit queries for in one request, default is 1

    Returns:
        Nothing returned yet, will return basic stats to help decide whether to use assay or not
    """
    ncmpds=len(myList)
    smiles_lst,cid_lst,inchikey_lst=[],[],[]
    sublst=""
    fail_lst=[]
    jsn_lst=[]
    for it in range(0,ncmpds,intv) :
        if (it+intv) > ncmpds :
            upbnd=ncmpds
        else :
            upbnd=it+intv
        sublst=myList[it:upbnd]
        inchikey = ','.join(map(str,sublst)) 
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+inchikey+"/summary/JSON"
        try :
            response = urlreq.urlopen(url)
            html = response.read()
        except :
            fail_lst.append(inchikey)
            continue
        f=io.BytesIO(html)
        cnt=0
        json_str=""
        for l in f :
            l=l.decode("utf-8") 
            l=l.rstrip()
            json_str += l
        jsn_lst.append(json_str)
    return jsn_lst
#    save_smiles_df=pd.DataFrame( {'CID' : cid_lst, 'standard_inchi_key' :inchikey_lst, 'smiles' : smiles_lst})
#    return save_smiles_df,fail_lst,discard_lst
     
#******************************************************************************************************************************************
def download_SID_from_bioactivity_assay(bioassayid) :
    """Retrieve summary info on bioactivity assays.

    Args:
        a single bioactivity id: PubChem AIDs (bioactivity assay ids)

    Returns:
        Returns the sids tested on this assay
    """
    myList=[bioassayid]
    ncmpds=len(myList)
    smiles_lst,cid_lst,inchikey_lst=[],[],[]
    sublst=""
    fail_lst=[]
    jsn_lst=[]
    intv=1
    for it in range(0,ncmpds,intv) :
        if (it+intv) > ncmpds :
            upbnd=ncmpds
        else :
            upbnd=it+intv
        sublst=myList[it:upbnd]
        inchikey = ','.join(map(str,sublst)) 
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+inchikey+"/sids/JSON"
        try :
            response = urlreq.urlopen(url)
            html = response.read()
        except :
            fail_lst.append(inchikey)
            continue
        f=io.BytesIO(html)
        cnt=0
        json_str=""
        for l in f :
            l=l.decode("utf-8") 
            l=l.rstrip()
            json_str += l
        jsn_lst.append(json_str)
    res=json.loads(jsn_lst[0])
    res_lst=res["InformationList"]['Information'][0]['SID']
    return res_lst
     
#https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/504526/doseresponse/CSV?sid=104169547,109967232

#******************************************************************************************************************************************
def download_dose_response_from_bioactivity(aid,sidlst) :
    """Retrieve data for assays for a select list of sids.

    Args:
        myList (list): a bioactivity id (aid)

        sidlst (list): list of sids specified as integers

    Returns:
        Nothing returned yet, will return basic stats to help decide whether to use assay or not
    """
    sidstr= "," . join(str(val) for val in sidlst)
    myList=[sidstr]
    ncmpds=len(myList)
    smiles_lst,cid_lst,inchikey_lst=[],[],[]
    sublst=""
    fail_lst=[]
    jsn_lst=[]
    intv=1
    for it in range(0,ncmpds,intv) :
        if (it+intv) > ncmpds :
            upbnd=ncmpds
        else :
            upbnd=it+intv
        sublst=myList[it:upbnd]
        inchikey = ','.join(map(str,sublst)) 
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+aid+"/doseresponse/CSV?sid="+inchikey
        try :
            response = urlreq.urlopen(url)
            html = response.read()
        except :
            fail_lst.append(inchikey)
            continue
        f=io.BytesIO(html)
        cnt=0
        json_str=""
        df=pd.read_csv(f)
        jsn_lst.append(df)
    return jsn_lst


#******************************************************************************************************************************************
def download_activitytype(aid,sid) :
    """Retrieve data for assays for a select list of sids.

    Args:
        myList (list): a bioactivity id (aid)

        sidlst (list): list of sids specified as integers

    Returns:
        Nothing returned yet, will return basic stats to help decide whether to use assay or not
    """
    myList=[sid]
    ncmpds=len(myList)
    smiles_lst,cid_lst,inchikey_lst=[],[],[]
    sublst=""
    fail_lst=[]
    jsn_lst=[]
    intv=1
    for it in range(0,ncmpds,intv) :
        if (it+intv) > ncmpds :
            upbnd=ncmpds
        else :
            upbnd=it+intv
        sublst=myList[it:upbnd]
        inchikey = ','.join(map(str,sublst)) 
        
        
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+aid+"/CSV?sid="+inchikey 
        #url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+aid+"/doseresponse/CSV?sid="+inchikey
        try :
            response = urlreq.urlopen(url)
            html = response.read()
        except :
            fail_lst.append(inchikey)
            continue
        f=io.BytesIO(html)
        cnt=0
        json_str=""
        df=pd.read_csv(f)
        jsn_lst.append(df)
    return jsn_lst
