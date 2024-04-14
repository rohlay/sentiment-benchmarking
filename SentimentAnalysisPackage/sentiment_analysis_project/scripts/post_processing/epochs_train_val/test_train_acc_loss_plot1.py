import matplotlib.pyplot as plt

# Your provided training accuracy and loss data
# MLP, SST2_2
#epochs:20
#train_accuracy= [0.5715759992599487, 0.6936458945274353, 0.7201828360557556, 0.7300604581832886, 0.7400854825973511, 0.741412341594696, 0.7462774515151978, 0.7629367709159851, 0.7543859481811523, 0.7629367709159851, 0.764853298664093, 0.7670647501945496, 0.7729617953300476, 0.7753206491470337, 0.7757629156112671, 0.7903582453727722, 0.7854931354522705, 0.7930119633674622, 0.7947810888290405, 0.803184449672699]
#train_loss = [0.6765849590301514, 0.5901678204536438, 0.5611525177955627, 0.5363845229148865, 0.5326774716377258, 0.5325126647949219, 0.519878625869751, 0.5099929571151733, 0.5111789703369141, 0.4987659752368927, 0.49473142623901367, 0.49005326628685, 0.48669302463531494, 0.4793452024459839, 0.472969114780426, 0.46013006567955017, 0.46356573700904846, 0.4493323564529419, 0.4482715129852295, 0.44558048248291016]

#epochs:30
#train_accuracy=  [0.5820433497428894, 0.6803774237632751, 0.7191508412361145, 0.7343358397483826, 0.7403803467750549, 0.7415598034858704, 0.7558602094650269, 0.7514374256134033, 0.7560076713562012, 0.760577917098999, 0.7604305148124695, 0.769865870475769, 0.774730920791626, 0.7720772624015808, 0.7819548845291138, 0.7809228897094727, 0.782692015171051, 0.7906531095504761, 0.7906531095504761, 0.798171877861023, 0.7999410033226013, 0.8046587109565735, 0.8012678623199463, 0.8099660873413086, 0.8112929463386536, 0.8179271817207336, 0.8207283020019531, 0.8170425891876221, 0.8192540407180786, 0.8309007883071899]
#train_loss = [0.6707592606544495, 0.6033399701118469, 0.5611286759376526, 0.5423340201377869, 0.5334416627883911, 0.5287584662437439, 0.5192408561706543, 0.5176000595092773, 0.5080270171165466, 0.4996313154697418, 0.4950968623161316, 0.4841149151325226, 0.4809979498386383, 0.48393329977989197, 0.4714048206806183, 0.469061940908432, 0.46650978922843933, 0.4576607346534729, 0.45003950595855713, 0.4479171931743622, 0.4403020441532135, 0.42999598383903503, 0.4343169629573822, 0.42148876190185547, 0.4163295328617096, 0.4110029637813568, 0.39940327405929565, 0.4091464877128601, 0.4010016918182373, 0.3854350447654724]

#epochs:35
#train_accuracy=  [0.558160126209259, 0.6780185699462891, 0.7148754000663757, 0.7343358397483826, 0.7464249134063721, 0.7473094463348389, 0.74568772315979, 0.7495208382606506, 0.7583665251731873, 0.7554179430007935, 0.7650007605552673, 0.7680966854095459, 0.7728143930435181, 0.7753206491470337, 0.7795960307121277, 0.7819548845291138, 0.7847560048103333, 0.7888839840888977, 0.7980244755744934, 0.7977296113967896, 0.8020049929618835, 0.8017101287841797, 0.8111454844474792, 0.8124723434448242, 0.8195488452911377, 0.8174849152565002, 0.823971688747406, 0.822644829750061, 0.8258882761001587, 0.8257408142089844, 0.8297213912010193, 0.8360607624053955, 0.8373875617980957, 0.8344390392303467, 0.8440217971801758]
#train_loss = [0.6831016540527344, 0.6005550026893616, 0.5628327131271362, 0.5491799712181091, 0.5308024883270264, 0.5263505578041077, 0.5217270851135254, 0.5134188532829285, 0.5035954713821411, 0.5012990832328796, 0.49830472469329834, 0.4913882613182068, 0.47601959109306335, 0.47154200077056885, 0.46860992908477783, 0.46654507517814636, 0.46081554889678955, 0.45579060912132263, 0.4513531029224396, 0.44015026092529297, 0.43963319063186646, 0.43193918466567993, 0.4198910892009735, 0.4104577302932739, 0.41320309042930603, 0.4058757424354553, 0.4011799991130829, 0.39952895045280457, 0.389123797416687, 0.388773113489151, 0.38171279430389404, 0.37234750390052795, 0.3683313727378845, 0.37679269909858704, 0.35628053545951843]

#epochs:50
train_accuracy=  [0.5714285969734192, 0.6870116591453552, 0.716939389705658, 0.7307975888252258, 0.7392009496688843, 0.7422969341278076, 0.7502579689025879, 0.7465723156929016, 0.7517322897911072, 0.7557128071784973, 0.7620521783828735, 0.7667698860168457, 0.7708978056907654, 0.7688338756561279, 0.7720772624015808, 0.7821022868156433, 0.7872622609138489, 0.7875571250915527, 0.7916851043701172, 0.7881468534469604, 0.7972873449325562, 0.801857590675354, 0.8012678623199463, 0.8049535751342773, 0.8118826746940613, 0.8171900510787964, 0.8146837949752808, 0.8185169100761414, 0.8213180303573608, 0.8195488452911377, 0.8283945322036743, 0.8283945322036743, 0.831932783126831, 0.8379772901535034, 0.8370927572250366, 0.8381247520446777, 0.8450537919998169, 0.8446115255355835, 0.8441692590713501, 0.8485920429229736, 0.8552262783050537, 0.8497714996337891, 0.8586171269416809, 0.8586171269416809, 0.8656936287879944, 0.8606811165809631, 0.8592068552970886, 0.8662833571434021, 0.8648090958595276, 0.8628925085067749]
train_loss = [0.6740703582763672, 0.5979512333869934, 0.5648117065429688, 0.5493581295013428, 0.5401610136032104, 0.5292030572891235, 0.5219621658325195, 0.5150787830352783, 0.5144456624984741, 0.5055160522460938, 0.4978511929512024, 0.4939975142478943, 0.4840329885482788, 0.481783002614975, 0.48029759526252747, 0.46875783801078796, 0.46517086029052734, 0.4581802785396576, 0.44886428117752075, 0.45325353741645813, 0.4402168393135071, 0.4343072175979614, 0.4389228820800781, 0.4240719676017761, 0.41769373416900635, 0.4089876711368561, 0.40976524353027344, 0.4030483067035675, 0.4006193280220032, 0.40191981196403503, 0.39062362909317017, 0.37884247303009033, 0.3782571852207184, 0.3703329563140869, 0.3767376244068146, 0.36554020643234253, 0.3535051643848419, 0.35571423172950745, 0.3599404990673065, 0.35115963220596313, 0.34499651193618774, 0.35078445076942444, 0.3334851562976837, 0.3354487419128418, 0.33055341243743896, 0.3230332136154175, 0.32311782240867615, 0.3190635144710541, 0.31583189964294434, 0.3214167058467865]


epochs = list(range(1, len(train_accuracy) + 1))  # Assuming epochs start at 1

# Create plots
fig, ax1 = plt.subplots()

# Plot training accuracy
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Accuracy', color=color)
ax1.plot(epochs, train_accuracy, color=color, marker='o')  # 'o' for points
ax1.tick_params(axis='y', labelcolor=color)

# Create another y-axis for the training loss using the same x-axis
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Train Loss', color=color)  
ax2.plot(epochs, train_loss, color=color, marker='o')  # 'o' for points
ax2.tick_params(axis='y', labelcolor=color)

# Title and other customizations
fig.tight_layout()
plt.title('Training Accuracy and Loss')
ax1.grid(True)

# Show the plot
plt.show()