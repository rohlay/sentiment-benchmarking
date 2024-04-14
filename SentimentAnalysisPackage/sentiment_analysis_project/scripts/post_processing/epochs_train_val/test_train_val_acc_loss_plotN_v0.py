import matplotlib.pyplot as plt

# Placeholder data - replace these with your actual lists of lists.
# Each sublist corresponds to data from a different epoch setting.
 # Add your training accuracies for each epoch setting
 # Add your training losses for each epoch setting
 # Add your validation accuracies for each epoch setting
 # Add your validation losses for each epoch setting

train_accuracies = [
    [0.5577178001403809, 0.6814094185829163, 0.7246056199073792, 0.7350729703903198, 0.7433289289474487],
    [0.5649417638778687, 0.6758071780204773, 0.7215096354484558, 0.7340409755706787, 0.7377266883850098, 0.7430340647697449, 0.7450980544090271, 0.7583665251731873, 0.7595459222793579, 0.7682441473007202],
    [0.5673006176948547, 0.6845054030418396, 0.720625102519989, 0.7375792264938354, 0.7366946935653687, 0.7465723156929016, 0.7484888434410095, 0.752027153968811, 0.763526439666748, 0.7627893090248108, 0.7703081369400024, 0.7694235444068909, 0.7708978056907654, 0.7726669907569885, 0.7865251302719116],
    [0.5724605917930603, 0.6850950717926025, 0.7212148308753967, 0.7322718501091003, 0.7417072057723999, 0.7490785717964172, 0.7529116868972778, 0.7586613297462463, 0.7570396661758423, 0.7616099119186401, 0.7726669907569885, 0.7723721265792847, 0.7757629156112671, 0.7772372364997864, 0.7891788482666016, 0.786967396736145, 0.7874097228050232, 0.7943387627601624, 0.7924222350120544, 0.7984667420387268],
    [0.5748193860054016, 0.6881910562515259, 0.7215096354484558, 0.7321244478225708, 0.7445083260536194, 0.7501105666160583, 0.751289963722229, 0.7610201835632324, 0.7596933245658875, 0.756302535533905, 0.7700132727622986, 0.7701606750488281, 0.774730920791626, 0.7807754874229431, 0.7790063619613647, 0.7866725921630859, 0.7933067679405212, 0.7866725921630859, 0.792569637298584, 0.7989090085029602, 0.8049535751342773, 0.8008255958557129, 0.8071649670600891, 0.8105558156967163, 0.8132094740867615, 0.8195488452911377, 0.8186643123626709, 0.8219076991081238, 0.8232345581054688, 0.8267728090286255],
    [0.5904467105865479, 0.6995429992675781, 0.7209199666976929, 0.7337461113929749, 0.7333038449287415, 0.7455403208732605, 0.7450980544090271, 0.7514374256134033, 0.7641161680221558, 0.7607253193855286, 0.769128680229187, 0.7713401317596436, 0.7734041213989258, 0.781070351600647, 0.779743492603302, 0.7922748327255249, 0.7912428379058838, 0.7989090085029602, 0.7913902401924133, 0.7977296113967896, 0.8021524548530579, 0.8096712231636047, 0.8136518001556396, 0.8158631920814514, 0.8152734637260437, 0.8208757042884827, 0.8232345581054688, 0.8303110599517822, 0.8303110599517822, 0.8339967727661133, 0.8372401595115662, 0.8376824259757996, 0.8382721543312073, 0.8426949977874756, 0.840483546257019, 0.8513931632041931, 0.8484446406364441, 0.8524251580238342, 0.846085786819458, 0.8544891476631165, 0.8614182472229004, 0.860975980758667, 0.8537520170211792, 0.858322262763977, 0.8645142316818237, 0.8679050803184509, 0.8727701902389526, 0.8673153519630432, 0.8648090958595276, 0.8645142316818237]


                     
] 
train_losses = [
    [0.6827008128166199, 0.602400004863739, 0.555899977684021, 0.5449792742729187, 0.5374887585639954],
    [0.682153582572937, 0.6046280860900879, 0.5557681918144226, 0.5450478792190552, 0.5398945808410645, 0.5325126051902771, 0.5245949625968933, 0.505061686038971, 0.504135012626648, 0.49655601382255554],
    [0.6768027544021606, 0.5986692905426025, 0.5582386255264282, 0.5455353260040283, 0.5391067862510681, 0.5234644412994385, 0.5185672640800476, 0.5122523903846741, 0.49945539236068726, 0.5038594603538513, 0.4940754175186157, 0.4845285415649414, 0.48458603024482727, 0.4797060191631317, 0.46455419063568115],
    [0.6788532733917236, 0.5977282524108887, 0.5613816976547241, 0.5435815453529358, 0.5349591374397278, 0.5256223678588867, 0.5205786228179932, 0.5137986540794373, 0.5056061744689941, 0.49511653184890747, 0.4910666346549988, 0.48409587144851685, 0.47645869851112366, 0.47743111848831177, 0.46669426560401917, 0.46376922726631165, 0.45726364850997925, 0.4509904384613037, 0.44583702087402344, 0.44216644763946533],
    [0.6734225749969482, 0.59688401222229, 0.5633789896965027, 0.542231559753418, 0.5330847501754761, 0.5256473422050476, 0.5230576395988464, 0.5086124539375305, 0.5040819644927979, 0.5023737549781799, 0.49396616220474243, 0.48631149530410767, 0.4837358295917511, 0.4720238745212555, 0.4726255536079407, 0.46147096157073975, 0.4542723000049591, 0.45391836762428284, 0.4499439299106598, 0.4408334791660309, 0.43661072850227356, 0.4334025979042053, 0.4256015419960022, 0.4255344569683075, 0.4143843948841095, 0.4120918810367584, 0.40783894062042236, 0.397137314081192, 0.39675483107566833, 0.3938685953617096],
    [0.6659937500953674, 0.5871905088424683, 0.5643793940544128, 0.5458270907402039, 0.5373318791389465, 0.5234659910202026, 0.5238845348358154, 0.5134044289588928, 0.5002250075340271, 0.4993680417537689, 0.4929734170436859, 0.4874641001224518, 0.4756670296192169, 0.4772298038005829, 0.4702453315258026, 0.4558511972427368, 0.4547262191772461, 0.44772398471832275, 0.4472425878047943, 0.43995150923728943, 0.43541836738586426, 0.421730101108551, 0.4161527454853058, 0.4107469320297241, 0.4098096489906311, 0.4043300747871399, 0.3992156386375427, 0.3911377489566803, 0.3775748312473297, 0.3835448920726776, 0.36946532130241394, 0.3727399706840515, 0.3692348599433899, 0.3693958520889282, 0.3669401705265045, 0.351949006319046, 0.35448652505874634, 0.343198299407959, 0.35111117362976074, 0.3432175815105438, 0.3283916711807251, 0.33480504155158997, 0.3361601233482361, 0.33088961243629456, 0.31915897130966187, 0.3222859501838684, 0.310949444770813, 0.3121418356895447, 0.3183666467666626, 0.3163013160228729]




]      
val_accuracies = [
    [0.7193396091461182, 0.7346698045730591, 0.7517688870429993, 0.7529481053352356, 0.7535377144813538],
    [0.7158018946647644, 0.7452830076217651, 0.7494103908538818, 0.7387971878051758, 0.7529481053352356, 0.7517688870429993, 0.7464622855186462, 0.7505896091461182, 0.7505896091461182, 0.7529481053352356],
    [0.7128537893295288, 0.7429245114326477, 0.7411556839942932, 0.7476415038108826, 0.7523584961891174, 0.760613203048706, 0.7564858198165894, 0.7547169923782349, 0.7576650977134705, 0.7623820900917053, 0.7594339847564697, 0.7582547068595886, 0.7529481053352356, 0.7629716992378235, 0.7535377144813538],
    [0.7152122855186462, 0.7417452931404114, 0.7541273832321167, 0.7476415038108826, 0.760613203048706, 0.7612028121948242, 0.7564858198165894, 0.7529481053352356, 0.7588443160057068, 0.7653301954269409, 0.7570754885673523, 0.7629716992378235, 0.7529481053352356, 0.7452830076217651, 0.7612028121948242, 0.7547169923782349, 0.7617924809455872, 0.7535377144813538, 0.7576650977134705, 0.7505896091461182],
    [0.7175707817077637, 0.739386796951294, 0.7441037893295288, 0.744693398475647, 0.7494103908538818, 0.7458726167678833, 0.7517688870429993, 0.7441037893295288, 0.7517688870429993, 0.7494103908538818, 0.7558962106704712, 0.7494103908538818, 0.7482311129570007, 0.7505896091461182, 0.7600235939025879, 0.7482311129570007, 0.7535377144813538, 0.7511792182922363, 0.7564858198165894, 0.7494103908538818, 0.7488207817077637, 0.7529481053352356, 0.7541273832321167, 0.7529481053352356, 0.7464622855186462, 0.7535377144813538, 0.760613203048706, 0.7505896091461182, 0.7476415038108826, 0.7558962106704712],
    [0.7305424809455872, 0.7411556839942932, 0.7488207817077637, 0.7423349022865295, 0.7523584961891174, 0.7494103908538818, 0.7535377144813538, 0.7529481053352356, 0.7594339847564697, 0.7488207817077637, 0.7523584961891174, 0.7535377144813538, 0.7564858198165894, 0.7588443160057068, 0.7523584961891174, 0.7529481053352356, 0.755306601524353, 0.7576650977134705, 0.7458726167678833, 0.7476415038108826, 0.7476415038108826, 0.7470518946647644, 0.744693398475647, 0.7464622855186462, 0.7370283007621765, 0.7576650977134705, 0.7588443160057068, 0.7547169923782349, 0.7476415038108826, 0.7464622855186462, 0.7452830076217651, 0.7476415038108826, 0.7435141801834106, 0.7458726167678833, 0.7417452931404114, 0.7494103908538818, 0.75, 0.7423349022865295, 0.7435141801834106, 0.7441037893295288, 0.7441037893295288, 0.7423349022865295, 0.7340801954269409, 0.7435141801834106, 0.7511792182922363, 0.7458726167678833, 0.7435141801834106, 0.7517688870429993, 0.7464622855186462, 0.7417452931404114]



    ]    

val_losses = [
    [0.6060424447059631, 0.5358201265335083, 0.5236091017723083, 0.5206471681594849, 0.5150008201599121],
    [0.6141059994697571, 0.5365738272666931, 0.525479793548584, 0.5224752426147461, 0.5172655582427979, 0.5142510533332825, 0.5127663016319275, 0.513843297958374, 0.5122523307800293, 0.5162490010261536],
    [0.6024328470230103, 0.5395463109016418, 0.5287937521934509, 0.5236427783966064, 0.5163961052894592, 0.5171312689781189, 0.5143287777900696, 0.5153529047966003, 0.5120114088058472, 0.5124961137771606, 0.5109334588050842, 0.5141598582267761, 0.5147954821586609, 0.5125686526298523, 0.5168646574020386],
    [0.6093864440917969, 0.5356391668319702, 0.5181245803833008, 0.5180732607841492, 0.5146002769470215, 0.5137877464294434, 0.515923023223877, 0.5169144868850708, 0.5145469903945923, 0.5084867477416992, 0.511734664440155, 0.512158215045929, 0.5117055177688599, 0.5192641019821167, 0.5147151350975037, 0.5142161846160889, 0.5111075639724731, 0.5190107226371765, 0.5186238288879395, 0.5172186493873596],
    [0.5808283090591431, 0.5346008539199829, 0.5279219150543213, 0.5183177590370178, 0.5201680660247803, 0.5136116743087769, 0.5132135152816772, 0.5148527026176453, 0.5176786184310913, 0.5129355788230896, 0.5137366652488708, 0.5148496627807617, 0.5279269814491272, 0.5196834206581116, 0.5151457786560059, 0.5213525295257568, 0.522221028804779, 0.5249608159065247, 0.5223301649093628, 0.536952555179596, 0.5301526784896851, 0.527377188205719, 0.5343946218490601, 0.5271995663642883, 0.5411555171012878, 0.538403332233429, 0.5534209609031677, 0.5540380477905273, 0.5481816530227661, 0.5541514158248901],
    [0.5828225612640381, 0.5351647138595581, 0.5243869423866272, 0.5176035165786743, 0.5210309624671936, 0.5134003758430481, 0.5132402181625366, 0.5149141550064087, 0.512072741985321, 0.5145241022109985, 0.513661801815033, 0.5150267481803894, 0.5135315656661987, 0.5134923458099365, 0.517257571220398, 0.5173589587211609, 0.5162770748138428, 0.5132025480270386, 0.5189979672431946, 0.530125081539154, 0.5209743976593018, 0.534800112247467, 0.5279932022094727, 0.5305728316307068, 0.5372322201728821, 0.5324082970619202, 0.5420182347297668, 0.5437147617340088, 0.5554319620132446, 0.5630783438682556, 0.5660939812660217, 0.5454720258712769, 0.5664111375808716, 0.5775402784347534, 0.5652470588684082, 0.560901403427124, 0.5773215293884277, 0.5852149128913879, 0.586971640586853, 0.589177131652832, 0.5879512429237366, 0.5859580039978027, 0.6055903434753418, 0.6001186966896057, 0.6016753911972046, 0.6028762459754944, 0.6309705972671509, 0.6140196919441223, 0.6133618354797363, 0.6280433535575867]




]     


#################


epoch_settings = [5, 10, 15,20,30,50]  # The different epoch settings you're using

# Create a figure and axes with a subplot for each pair of train/val data
fig, axes = plt.subplots(len(epoch_settings), 2, figsize=(10, len(epoch_settings) * 5))  # 5 for the height of each subplot

for i, (epochs, train_acc, train_loss, val_acc, val_loss) in enumerate(zip(epoch_settings, train_accuracies, train_losses, val_accuracies, val_losses)):
    # Adjust the range of epochs if each setting has a different range
    epochs_range = list(range(1, epochs + 1))

    # Training subplot
    ax1 = axes[i, 0]
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Accuracy', color=color)
    ax1.plot(epochs_range, train_acc, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Train Loss', color=color)
    ax2.plot(epochs_range, train_loss, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'Training - Epochs: {epochs}')
    ax1.grid(True)

    # Validation subplot
    ax3 = axes[i, 1]
    color = 'tab:green'
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Validation Accuracy', color=color)
    ax3.plot(epochs_range, val_acc, color=color, marker='o')
    ax3.tick_params(axis='y', labelcolor=color)
    ax4 = ax3.twinx()
    color = 'tab:purple'
    ax4.set_ylabel('Validation Loss', color=color)
    ax4.plot(epochs_range, val_loss, color=color, marker='o')
    ax4.tick_params(axis='y', labelcolor=color)
    ax3.set_title(f'Validation - Epochs: {epochs}')
    ax3.grid(True)

# Adjust the layout
plt.tight_layout()
plt.show()
