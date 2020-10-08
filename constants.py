core_numbers = 20

query = """
SELECT 
  DIM.PLNT_CD,                                                                                  # PLANT CODE
  DIM.MTRL_CD,                                                                                  # MATERIAL CODE
  DIM_MTRL.MTRL_PROD_HIERCY_4,                                                                  # SUB-TYPE MATERIAL GROUP
  DIM_MTRL.MTRL_LEN,                                                                            # MATERIAL LENGTH
  DIM_MTRL.MTRL_WIDTH,                                                                          # MATERIAL WIDTH
  DIM_MTRL.MTRL_HGT,                                                                            # MATERIAL HEIGTH
  DIM_MTRL.GRS_WGT,                                                                             # GROSS WEIGTH
  FT_PRC.PRC_SLS_VKP0,                                                                          # PRODUCT'S SALES PRICE
  DIM_MTRL.CUST_PRJCT,                                                                          # CUSTOMER PROJECT
  CAST(CASE WHEN DIM_MTRL.INTL_PRCHS = 'X' THEN 1 ELSE 0 END AS INT64) as INTL_PRCHS,           # INTERNATIONAL PURCHASE
  CAST(CASE WHEN DIM_MTRL.MTRL_MDD = 'X' THEN 1 ELSE 0 END AS INT64) as MTRL_MDD,               # INTERNAL PRODUCED MATERIAL 
FROM
  `brlm-web-data.dwh.DIM_MTRL_PLNT` DIM
  LEFT JOIN `brlm-web-data.dwh.FT_PRC_MTRL_PLNT` FT_PRC ON
    DIM.PLNT_CD = FT_PRC.PLNT_CD
    AND DIM.MTRL_CD = FT_PRC.MTRL_CD
  LEFT JOIN `brlm-web-data.dwh.DIM_MTRL` DIM_MTRL ON
    DIM.MTRL_CD = DIM_MTRL.MTRL_CD
  LEFT JOIN (SELECT PLNT_CD, MTRL_CD, min(SLS_DT) FRST_SLS FROM `brlm-web-data.dwh.FT_SLS` group by PLNT_CD, MTRL_CD) FT_FRST_SLS  ON
    DIM.PLNT_CD = CAST(FT_FRST_SLS.PLNT_CD AS STRING)
    AND DIM.MTRL_CD = CAST(FT_FRST_SLS.MTRL_CD AS STRING)
  LEFT JOIN `brlm-web-data.dwh.DIM_PLNT` DIM_PLNT ON
      DIM.PLNT_CD = DIM_PLNT.PLNT_CD
WHERE
  FT_PRC.PRC_DT = DATE_ADD(((SELECT CURRENT_DATE)), INTERVAL -1 DAY)
  AND FT_FRST_SLS.FRST_SLS <= DATE_ADD(((SELECT CURRENT_DATE)), INTERVAL -60 DAY)
  AND DIM_MTRL.MTRL_TYP = 'ZREV'
  AND FT_PRC.PRC_SLS_VKP0 > 0
  AND DIM_MTRL.MTRL_LEN > 0
  AND DIM_MTRL.MTRL_WIDTH > 0
  AND DIM_MTRL.MTRL_HGT > 0
  AND DIM_MTRL.GRS_WGT > 0
  AND DIM_PLNT.DSTRCT_SLS NOT IN ('6', '99')
  AND DIM_MTRL.MTRL_PROD_HIERCY_4 IS NOT NULL
"""

dict_translate_new = {
    'Centro' : 'PLNT_CD',
    'Material' : 'MTRL_CD',
    'Seção' : 'PRCHS_GRP',
    'Comprimento' : 'MTRL_LEN',
    'Largura' : 'MTRL_WIDTH',
    'Altura' : 'MTRL_HGT',
    'Peso' : 'GRS_WGT',
    'Preço de Venda' : 'PRC_SLS_VKP0',
    'Projeto Cliente' : 'CUST_PRJCT',
    'Imporato' : 'INTL_PRCHS',
    'MDH' : 'MTRL_MDD',}

dict_translate_export = {
    'Centro' : 'PLNT_CD',
    'Material' : 'MTRL_CD',
    'Seção' : 'PRCHS_GRP',
    'Comprimento' : 'MTRL_LEN',
    'Largura' : 'MTRL_WIDTH',
    'Altura' : 'MTRL_HGT',
    'Peso' : 'GRS_WGT',
    'Preço de Venda' : 'PRC_SLS_VKP0',
    'Projeto Cliente' : 'CUST_PRJCT',
    'Imporato' : 'INTL_PRCHS',
    'MDH' : 'MTRL_MDD',}