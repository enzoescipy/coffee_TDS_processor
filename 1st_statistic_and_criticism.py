import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spline
import scipy.stats as stats
from pprint import pprint
import math

#data refinement
class Dataset:
    def __init__(self, datalist):
        expr = self._refine(datalist)
        if expr == -1 :
            self.err = "ERROR!"
            return 
        self.id = expr["id"]
        self.expr_num, self.trial_num, self.repeat_num = map(int,list(self.id))
        self.vapo_conc = ((expr["bowl_after"] - expr["bowl_blank"]) / (expr["bowl_before"] - expr["bowl_blank"]))*100
        self.dillution_rate = expr["source_mass"] / (expr["water_mass"] + expr["source_mass"])
        self.TDS = expr["mixed_TDS"] / 10000
        self.EC = expr["mixed_EC"] / 10000
        self.Brix = expr["mixed_Brix"]
        self.temp = expr["mixed_temp"]

    def _refine(self, datalist):
        namelist = ["id","X","source_mass","X","water_mass","X","water_temp","mixed_temp","mixed_TDS","mixed_EC","mixed_Brix","bowl_blank","bowl_before","bowl_after"]
        datadict = {}
        try:
            datadict = dict(zip(namelist, datalist))
            while True:
                checkif = datadict.pop("X",None)
                if checkif == None:
                    break
            numberized_dict = {}
            for key, value in datadict.items():
                if key == "id":
                    numberized_dict[key] = value
                    continue
                numberized_dict[key] = float(value)
            return numberized_dict
        except Exception as exp:
            print(exp)
            return -1
        

datasets = []

with open("DATA.csv", "r") as f:
    while True:
        line = f.readline()
        lineparsed = line.strip("\n").split(",")
        if line == "":
            break
        datasets.append(Dataset(lineparsed))


expr1 = [[],[],[]]
expr2 = [[],[],[]]

for dataset in datasets:
    if dataset.expr_num == 1:
        expr1[dataset.repeat_num - 1].append(dataset)
    elif dataset.expr_num == 2:
        expr2[dataset.repeat_num - 1].append(dataset)


#expr2 : deviation calculate
expr2_SD = []
for trial in expr2:
    print("2!")
    avr_vapoConc = 0
    avr_TDS = 0
    avr_EC = 0
    avr_Brix = 0
    for data in trial:
        avr_vapoConc += data.vapo_conc
        avr_TDS += data.TDS
        avr_EC += data.EC
        avr_Brix += data.Brix
    avr_vapoConc /= len(trial) 
    avr_TDS /= len(trial) 
    avr_EC /= len(trial) 
    avr_Brix /= len(trial) 

    sD_vapoConc = 0
    sD_TDS = 0
    sD_EC = 0
    sD_Brix = 0
    for data in trial:
        sD_vapoConc += (avr_vapoConc - data.vapo_conc)**2
        sD_TDS += (avr_TDS - data.TDS)**2
        sD_EC += (avr_EC - data.EC)**2
        sD_Brix += (avr_Brix - data.Brix)**2
    
    sD_vapoConc = math.sqrt(sD_vapoConc)
    sD_TDS = math.sqrt(sD_TDS)
    sD_EC = math.sqrt(sD_EC)
    sD_Brix = math.sqrt(sD_Brix)
    expr2_SD.append([(sD_vapoConc,avr_vapoConc),
                    (sD_TDS,avr_TDS),
                    (sD_EC,avr_EC),
                    (sD_Brix,avr_Brix)])

#expr1 : linearlity calculate
expr1_lty = []
for trial in expr1:
    print("1!")
    #function splinizing
    basicfunc_dict = {"x_dilluRate":[],"y_vapoConc":[],"y_TDS":[],"y_EC":[],"y_Brix":[]}

    for data in trial:
        basicfunc_dict["x_dilluRate"].append(data.dillution_rate)
        basicfunc_dict["y_vapoConc"].append(data.vapo_conc)
        basicfunc_dict["y_TDS"].append(data.TDS)
        basicfunc_dict["y_EC"].append(data.EC)
        basicfunc_dict["y_Brix"].append(data.Brix)

    np_basicfunc_dict = {}
    for key, value in basicfunc_dict.items():
        np_basicfunc_dict[key] = np.array(list(reversed(basicfunc_dict[key])))
    np_spliedPrefunc_dict = {
        "vapoConc":spline.interp1d(basicfunc_dict["x_dilluRate"],basicfunc_dict["y_vapoConc"],kind='quadratic'),
        "TDS":spline.interp1d(basicfunc_dict["x_dilluRate"],basicfunc_dict["y_TDS"],kind='quadratic'),
        "EC":spline.interp1d(basicfunc_dict["x_dilluRate"],basicfunc_dict["y_EC"],kind='quadratic'),
        "Brix":spline.interp1d(basicfunc_dict["x_dilluRate"],basicfunc_dict["y_Brix"],kind='quadratic')
    }

    np_spliedfunc_dict = {}

    dilluRate_linspace = np.linspace(basicfunc_dict["x_dilluRate"][0],basicfunc_dict["x_dilluRate"][-1],102)
    for key,value in np_spliedPrefunc_dict.items():
        np_spliedfunc_dict[key] = np_spliedPrefunc_dict[key](dilluRate_linspace)

    #function double differential

    def derivative(index, xarr, yarr):
        xb = xarr[index-1]
        xf = xarr[index+1]
        yb = yarr[index-1]
        yf = yarr[index+1]
        return (yf-yb)/(xf-xb)

    np_diff1Func_dict = {}
    np_diff1avg_dict = {}

    for key, value in np_spliedfunc_dict.items():
        target_funcArr = value
        result = []
        diff_avg = 0
        for i in range(1, len(target_funcArr) - 1):
            output = derivative(i,dilluRate_linspace,target_funcArr)
            diff_avg += output
            result.append(output)
        diff_avg /= (len(target_funcArr) - 1)
        np_diff1avg_dict[key] = diff_avg
        np_diff1Func_dict[key] = np.array(result)
    
    dilluRate_Diff1_linspace = []
    for i in range(1, len(dilluRate_linspace)-1):
        dilluRate_Diff1_linspace.append(dilluRate_linspace[i])
    dilluRate_Diff1_linspace = np.array(dilluRate_Diff1_linspace)




    np_diff2Func_dict = {}

    for key, value in np_diff1Func_dict.items():
        target_funcArr = value
        result = []
        for i in range(1, len(target_funcArr) - 1):
            output = derivative(i,dilluRate_Diff1_linspace,target_funcArr)
            result.append(output)
        np_diff2Func_dict[key] = np.array(result)
    
    dilluRate_Diff2_linspace = []
    for i in range(1, len(target_funcArr) - 1):
        dilluRate_Diff2_linspace.append(dilluRate_Diff1_linspace[i])
    dilluRate_Diff2_linspace = np.array(dilluRate_Diff2_linspace)

    
    #function integral calculate

    def fxdx(index, xarr, yarr):
        dx = xarr[index+1] - xarr[index]
        dy = (yarr[index+1] + yarr[index]) / 2
        return dx * dy

    diff2_integral_dict = {}
    for key, value in np_diff2Func_dict.items():
        target_funcArr = value
        integral_sum = 0
        for i in range(len(target_funcArr) - 1):
            integral_sum += np.abs(fxdx(i,dilluRate_Diff2_linspace,target_funcArr))
        
        diff2_integral_dict[key] = integral_sum

    expr1_lty.append([(diff2_integral_dict["vapoConc"],np_diff1avg_dict["vapoConc"]), 
                    (diff2_integral_dict["TDS"],np_diff1avg_dict["TDS"]), 
                    (diff2_integral_dict["EC"],np_diff1avg_dict["EC"]), 
                    (diff2_integral_dict["Brix"],np_diff1avg_dict["Brix"])
                    ])
    
    '''
    plt.plot(dilluRate_linspace, np_spliedfunc_dict["vapoConc"],'r')
    plt.plot(dilluRate_linspace, np_spliedfunc_dict["TDS"],'g')
    plt.plot(dilluRate_linspace, np_spliedfunc_dict["Brix"],'b')

    plt.plot(dilluRate_Diff1_linspace, np_diff1Func_dict["vapoConc"],color='r', linestyle='--')
    plt.plot(dilluRate_Diff1_linspace, np_diff1Func_dict["TDS"],color='g', linestyle='--')
    plt.plot(dilluRate_Diff1_linspace, np_diff1Func_dict["Brix"],color='b', linestyle='--')

    plt.show()
    '''
print("vapor, TDS, EC, Brix")
print("1:concentration linearlity")
pprint(expr1_lty)
print("\n")
print("2:temp deviation")
pprint(expr2_SD)

#point distribution graph
##organize data
expr1_distributy_x = []
expr1_distributy_y = []
for chart in expr1_lty:
    for spot in chart:
        expr1_distributy_x.append(spot[1]) # size of gradient
        expr1_distributy_y.append(spot[0]) # concentration linearlity

expr2_distributy_x = []
expr2_distributy_y = []
for chart in expr2_SD:
    for spot in chart:
        expr2_distributy_x.append(spot[1]) # size of value
        expr2_distributy_y.append(spot[0]) # S.D of value in temp variation


expr1_distributy_x = np.array(expr1_distributy_x)
expr1_distributy_y = np.array(expr1_distributy_y)
expr2_distributy_x = np.array(expr2_distributy_x)
expr2_distributy_y = np.array(expr2_distributy_y)

'''
plt.subplot(2,1,1)
plt.scatter(expr1_distributy_x,expr1_distributy_y)

plt.subplot(2,1,2)
plt.scatter(expr2_distributy_x,expr2_distributy_y)

plt.show()
'''

#relation constant calculation
expr1_xySum = 0
expr1_xxSum = 0
expr1_yySum = 0
expr1_xSum = 0
expr1_ySum = 0

for index, vec in enumerate(list(zip(expr1_distributy_x, expr1_distributy_y))):
    x = vec[0]
    y = vec[1]
    expr1_xySum += x*y
    expr1_xxSum += x*x
    expr1_yySum += y*y
    expr1_xSum += x
    expr1_ySum += y

expr1_len = len(expr1_distributy_x)

expr1_Sxy = expr1_xySum - ( (expr1_xSum*expr1_ySum) / (expr1_len) )
expr1_Sxx = expr1_xxSum - ( (expr1_xSum)**2 / (expr1_len) )
expr1_Syy = expr1_yySum - ( (expr1_ySum)**2 / (expr1_len) )

expr1_R = expr1_Sxy / math.sqrt(expr1_Sxx*expr1_Syy)

expr2_xySum = 0
expr2_xxSum = 0
expr2_yySum = 0
expr2_xSum = 0
expr2_ySum = 0

for index, vec in enumerate(list(zip(expr2_distributy_x, expr2_distributy_y))):
    x = vec[0]
    y = vec[1]
    expr2_xySum += x*y
    expr2_xxSum += x*x
    expr2_yySum += y*y
    expr2_xSum += x
    expr2_ySum += y

expr2_len = len(expr2_distributy_x)

expr2_Sxy = expr2_xySum - ( (expr2_xSum*expr2_ySum) / (expr2_len) )
expr2_Sxx = expr2_xxSum - ( (expr2_xSum)**2 / (expr2_len) )
expr2_Syy = expr2_yySum - ( (expr2_ySum)**2 / (expr2_len) )

expr2_R = expr2_Sxy / math.sqrt(expr2_Sxx*expr2_Syy)

print("expr1 relation const. = ", expr1_R)
print("expr2 relation const. = ", expr2_R)

#test statistic calc.
alpha = 0.5

expr1_statistic = expr1_R / math.sqrt( (1-(expr1_R)**2) / (expr1_len - 2) )
expr1_tFunc = stats.t(df = expr1_len-2).ppf
expr1_boundary = expr1_tFunc(1-alpha/2)
print("\nexpr1:")
print("boundary : +-",expr1_boundary, "\nstatistic =", expr1_statistic, "\n null hypothesis is :", not(expr1_boundary<expr1_statistic))

expr2_statistic = expr2_R / math.sqrt( (2-(expr2_R)**2) / (expr2_len - 2) )
expr2_tFunc = stats.t(df = expr2_len-2).ppf
expr2_boundary = expr2_tFunc(1-alpha/2)
print("\nexpr2:")
print("boundary : +-",expr2_boundary, "\nstatistic =", expr2_statistic, "\n null hypothesis is :", not(expr2_boundary<expr2_statistic))





    



