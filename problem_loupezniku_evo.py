# problém rozdělení množiny na k podmnožin se stejným součtem prvků
# pomocí evolučního algoritmu

# kódování jedince: vektor čísel {1,...,k} délky n (n je počet prvků množiny)
# chceme minimalizovat rozdíl mezi největším a nejmenším součtem prvků v podmnožinách
# chceme ruletu, problém se zápornou fitness
# fitness: 1/(max - min + 1) (čím menší rozdíl, tím větší fitness)

# případně můžeme použít turnajovou selekci
# nezáleží na škálování fitness, protože porovnáváme jen relativní hodnoty (pořadí jedinců)

