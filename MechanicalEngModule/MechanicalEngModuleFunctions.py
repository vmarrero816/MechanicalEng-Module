import numpy as np

###  Constants  ####

StefanBoltzmann = 5.6703e-8 # [W/(m^2 K^4)] stefan-Boltzmann 


### Definitions/Dictionaries  #####

def BoltSize(BoltNumber,BoltType):
    # bolt diameter [in],pitch, bolt head diameter [in]; min, bolt head height [in]; min
    match BoltNumber:
        case "#0":
            if BoltType=='Flat Head':
                pass
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .06, "Pitch":80, "Head Diameter": .091, "Head Height":.057}
            elif BoltType=='Flat Head 100 deg':   
                pass
        case "#1":
            if BoltType=='Flat Head':
                pass
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .073, "Pitch":64, "Head Diameter": .112, "Head Height":.070}
            elif BoltType=='Flat Head 100 deg':   
                pass   

        case "#2":
            if BoltType=='Flat Head':
                Bolt = {"Diameter" : .086, "Pitch":56, "Head Diameter": .126, "Head Height":.044}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .086, "Pitch":56, "Head Diameter": .134, "Head Height":.083}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .086, "Pitch":56, "Head Diameter": .126, "Head Height":.036}

        case "#4":
            if BoltType=='Flat Head':
                Bolt = {"Diameter" : .112, "Pitch":40, "Head Diameter": .168, "Head Height":.059}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .112, "Pitch":40, "Head Diameter": .176, "Head Height":.108}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .112, "Pitch":40, "Head Diameter": .177, "Head Height":.045}

        case "#6":
            if BoltType=='Flat Head':
                Bolt = {"Diameter" : .138, "Pitch":32, "Head Diameter": .211, "Head Height":.074}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .138, "Pitch":32, "Head Diameter": .218, "Head Height":.134}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .138, "Pitch":32, "Head Diameter": .231, "Head Height":.057}

        case "#8":
            if BoltType=='Flat Head':
                Bolt = {"Diameter" : .164, "Pitch":32, "Head Diameter": .255, "Head Height":.103}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .164, "Pitch":32, "Head Diameter": .262, "Head Height":.159}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .164, "Pitch":32, "Head Diameter": .270, "Head Height":.068}

        case "#10":
            if BoltType=='Flat Head':
                Bolt = {"Diameter" : .190, "Pitch":24, "Head Diameter": .296, "Head Height":.103}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .190, "Pitch":24, "Head Diameter": .303, "Head Height":.185}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .190, "Pitch":24, "Head Diameter": .322, "Head Height":.080}
            
        case "1/4":
            if BoltType=='Flat Head':
                Bolt = {"Diameter" : .25, "Pitch":20, "Head Diameter": .375, "Head Height":.133}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .25, "Pitch":20, "Head Diameter": .365, "Head Height":.244}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .25, "Pitch":20, "Head Diameter": .442, "Head Height":.106}

        case "3/8":
            if BoltType=='Flat Fillister Head':
                Bolt = {"Diameter" : .375, "Pitch":16, "Head Diameter": .536, "Head Height":.191}
            elif BoltType=='Socket Cap Head':
                Bolt = {"Diameter" : .375, "Pitch":16, "Head Diameter": .550, "Head Height":.368}
            elif BoltType=='Flat Head 100 deg':
                Bolt = {"Diameter" : .375, "Pitch":16, "Head Diameter": .694, "Head Height":.160}
            
    return Bolt


def MaterialDefinition(Material):
    match Material:
        case 'A286':
            MaterialProperties = {"Type" : 'Metal', 
                                  "Ultimate tensile strength" : 160e3, 
                                  "Yield strength" : 120e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :2.88e7, 
                                  "Linear CTE":9.17e-6,
                                  "Compressive Strength" : 0}
        case 'AL6061':
            MaterialProperties = {"Type" : 'Metal', 
                                  "Ultimate tensile strength" : 42e3, 
                                  "Yield strength" : 35e3,
                                  "Ultimate shear strength":27e3, 
                                  "Young modulus" :1e7, 
                                  "Linear CTE":13.1e-6,
                                  "Compressive Strength" : 0}            

        case 'CRES':
            MaterialProperties = {"Type" : 'Metal', 
                                  "Ultimate tensile strength" : 75e3, 
                                  "Yield strength" : 30e3,
                                  "Ultimate shear strength":48e3, 
                                  "Young modulus" :2.9e7, 
                                  "Linear CTE":8.9e-6,
                                  "Compressive Strength" : 0}

        case 'SS430':
            MaterialProperties = {"Type" : 'Metal', 
                                  "Ultimate tensile strength" : 40e3, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :2.7e6, 
                                  "Linear CTE":5.5e-6,
                                  "Compressive Strength" : 0}

        case 'G10':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 15e3, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :6e5, 
                                  "Linear CTEn":2.6e-5,
                                  "Compressive Strength" : 0e3}
        case 'PEEK':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 1.5e3, 
                                  "Yield strength" : 1.31e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :7.2e4, 
                                  "Linear CTE":8.9e-5,
                                  "Compressive Strength" : 0}
            
        case 'PTFE':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 3.9e3, 
                                  "Yield strength" : 3.4e3,
                                  "Ultimate shear strength":2e3, 
                                  "Young modulus" :2.9e7, 
                                  "Linear CTE":5.78e-6,
                                  "Compressive Strength" : 0}

        case 'Nylon12':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 6e3, 
                                  "Yield strength" : 0,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :2.39e5, 
                                  "Linear CTE":0,
                                  "Compressive Strength" : 0}
        case 'Torlon 5030':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 23e3, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :7.0e5, 
                                  "Linear CTE":0.9e-5,
                                  "Compressive Strength" : 40e3}

        case 'RO3206':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 7.9e3, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0e3, 
                                  "Young modulus" :462e6, 
                                  "Linear CTE":1.9e-5,
                                  "Compressive Strength" : 0}
            
        case 'Alumina98':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 348e3, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :0, 
                                  "Linear CTE":4.5e-6,
                                  "Compressive Strength" : 363e3}

        case 'RO4003C':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 18.9e6, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :2.85e6, 
                                  "Linear CTE":1.8e-5,
                                  "Compressive Strength" : 18.9e6}

        case 'RO4550F':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 18.9e6, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :2053e3, 
                                  "Linear CTE":1.8e-5,
                                  "Compressive Strength" : 18.9e6}
            
        case 'Arlon85N':
            MaterialProperties = {"Type" : 'Dielectric',
                                  "Ultimate tensile strength":48e3, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :3.6e6, 
                                  "Linear CTE":3e-5,
                                  "Compressive Strength" : 48e3}

        case 'Cho-seal 1285':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 200, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :1.25e3, 
                                  "Linear CTE":1.1e-4,
                                  "Compressive Strength" : 0e6}

        case 'SSP2368-65':
            MaterialProperties = {"Type" : 'Dielectric', 
                                  "Ultimate tensile strength" : 200, 
                                  "Yield strength" : 0e3,
                                  "Ultimate shear strength":0, 
                                  "Young modulus" :1.345e3, 
                                  "Linear CTE":1.1e-4,
                                  "Compressive Strength" : 0e6}
                
        case 'None':
            MaterialProperties = []
    return MaterialProperties

NutFactor = {'Dry': .3, 'Lubricated': .18,'Arathane': .22}


def InsertSize(InsertNumber):
    # Outside diameter [in] [Min,Max];
    match InsertNumber:
        case "#2":
            insert = {'MinDiameter':.110, 'MaxDiameter':.119} 
        case "#4":
            insert = {'MinDiameter':.144, 'MaxDiameter':.159}
        case "#6":
            insert = {'MinDiameter':.178, 'MaxDiameter':.193}
        case "#8":
            insert = {'MinDiameter':.205, 'MaxDiameter':.220}
        case "#10":
            insert = {'MinDiameter':.244, 'MaxDiameter':.259}
        case "1/4":
            insert = {'MinDiameter':.310, 'MaxDiameter':.330}
        case "3/8":
            insert = {'MinDiameter':.452, 'MaxDiameter':.472}
    return insert    
