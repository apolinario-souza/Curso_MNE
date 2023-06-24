
import numpy as np
import pandas as pd
import mne
import math


def criar_eventos(raw,time_sec_menosT0, n_tt):
	stim_info = mne.create_info(['eventos'], raw.info['sfreq'], ch_types=['stim'])

	events = np.zeros(((n_tt-1)*2,3))

	cont1  = 0
	cont2  = 0
	for i in range ((n_tt-1)*2):
    
    		if i % 2 == 0:    
        		events[i,:] = [int(time_sec_menosT0[cont1,0]*raw.info['sfreq']),0,i+1]
        		cont1+=1
    		else:
        		events[i,:] = [int(time_sec_menosT0[cont2,-1]*raw.info['sfreq']),0,i+1]
        		cont2+=1
    
        



	events = events.astype(np.int64) 



	stim_data = np.zeros((1, raw.n_times))

	stim_data[0, events[:, 0]] = events[:, 2]
	stim_raw = mne.io.RawArray(stim_data, stim_info)
	raw.add_channels([stim_raw], force_update_info=True)
	
	return raw


def convert_time_sec (dados_comp, n_tt, T0, ref_col1,ref_col2):
    time_sec = np.zeros((n_tt,2)) # Gerar uma variável que receberá os dados convertidos

    for tt in np.arange(n_tt): 
        
        t1 = sep_horas (dados_comp.iloc [tt,ref_col1])
        t2 = sep_horas (dados_comp.iloc [tt,ref_col2])
        
        time_sec [tt:, ] = time_to_sec(t1),time_to_sec(t2)

   
    time_sec_menosT0 = time_sec - T0
   
    
    return time_sec_menosT0



    
def sep_horas(lst): #colocar o ":" separandos as horas dos minutos e segundo
    
    lst = str(lst)
    lst = lst.replace('.', '')
    if float (lst[0]) <=2:
        returned= lst[0:2]+':'+lst[2:4]+':'+lst[4:6]+':'+lst[6:]
    else:
        returned = str(0)+lst[0:1]+':'+lst[1:3]+':'+lst[3:5]+':'+lst[5:]          
        
        
    return returned

def time_to_sec(t):
   h, m, s, ms = map(int, t.split(':'))
   return h * 3600 + m * 60 + s + (ms/1000)

def transf_virgula_ponto(dados_comp):
    dados_comp = np.array(dados_comp)
    lista_float = np.zeros(dados_comp.shape)

    for lin in range(dados_comp.shape[0]):    
        for col in range(dados_comp.shape[1]):
            try:
                if math.isnan(dados_comp[lin,col]): 
                    lista_float[lin,col]= float('nan')
                else:
                    lista_float[lin,col]= float(dados_comp[lin,col].replace(',', '.'))
            except:
                lista_float[lin,col]= float(dados_comp[lin,col].replace(',', '.'))
    return lista_float
    

def selecionar_tentativas(signal,eventos, n_tt, fs):
    a = []
    for i in range (n_tt):
        iniciar,stop = eventos[i,0], eventos[i,1]
            
        start = int(iniciar*fs)
        stop = int(stop*fs)
            
        a.append(np.nanmean(signal[start:stop])) #todas tentativas
            
   
    return np.array(a)

def selecionar_intervalos(signal,eventos, fs):
    a = []
    
    iniciar,stop = eventos[0], eventos[1]
            
    start = int(iniciar*fs)
    stop = int(stop*fs)
            
    a = signal[:,start:stop]
            
   
    return np.array(a)
