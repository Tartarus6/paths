import numpy
from random import random, randint
import matplotlib.pyplot as plt
from time import time, sleep
from matplotlib.animation import FuncAnimation
import imageio
# note: "node" refers to a position on the maze
# note: the value of a node is the cost that it takes to get there, so lower cost is better
# note: any code with "# non-resilient", and probably some more, will need to be changed for any major change in the pathfinding such as adding walls or changing the maze shape


# written based off of the pseudocode at "https://en.wikipedia.org/wiki/A*_search_algorithm"


def make_gif(frame_folder, fps, num_frames):
    print("gif")
    
    giffile = 'gif.gif'
    
    images_data = []
    for i in range(num_frames):
        data = imageio.imread(f'gif_folder/frame_{i}.jpg')
        images_data.append(data)
    
    imageio.mimwrite(giffile, images_data, format='.gif', fps=fps)


def grid_print(grid):  # non-resilient
    for i in grid:
        print(i)


def unzip(lst):  # yields the contents of the input list, removing nested lists. Output needs to be converted into a list to be used.
    for i in lst:
        if type(i) is list:
            yield from list(unzip(i))
        else:
            yield i


def make_maze(x: int, y: int, value_range: int, wall_rate: float, starting_position: tuple, ending_position: tuple):
    maze = [[0 for i in range(x)] for j in range(y)]  # initialising array of zeros

    # adding values to the array
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            maze[i][j] = randint(10, 10 + value_range)  # adding cost values
            if random() < wall_rate and ((i, j) != starting_position and (i, j) != ending_position):
                maze[i][j] = -1

    return maze


def manual_maze(input):
    l = input.split('\n')

    for i in range(len(l)):
        l[i] = [int(i) for i in l[i]]

    return l


def a_star(maze, starting_position: tuple, goal_position: tuple):
    print("run")
    
    
    class Output:
        def __init__(self, maze=None, final_path=None, failed=False, to_do_history=None, path_history=None, explored_history=None):
            if to_do_history is None:
                to_do_history = []
            self.maze = maze
            self.final_path = final_path
            self.failed = failed
            self.to_do_history = to_do_history
            self.path_history = path_history
            self.explored_history = explored_history
            

    def heuristic(current_position, end_position):  # estimates the cost to reach goal from the node at the given position
        # non-resilient
        
        # get average value of a node
        unzipped_maze = list(unzip(maze))
        # average_value = sum(list(unzipped_maze)) / len(unzipped_maze)

        average_value = 2.5
        
        # estimating moves needed to get to goal
        # assumes no walls or obstructions
        distance_to_goal = abs(max(current_position[0], end_position[0]) - min(current_position[0], end_position[0])) + abs(max(current_position[1], end_position[1]) - min(current_position[1], end_position[1]))
        
        return distance_to_goal * average_value
    
    def neighbors(position):  # finding available positions to move into from the given position
        # non-resilient
        
        output = []

        for i in range(-1, 2):  # y offset  # non-resilient
            for j in range(-1, 2):  # x offset  # non-resilient
                if i == 0 and j == 0:  # makes it so that the input position won't be added to the output as one of the neighbors and won't be outside the maze  # non-resilient
                    continue
                if (position[0] + i < 0) or (position[1] + j < 0) or (position[0] + i >= len(maze)) or (position[1] + j >= len(maze[i])):  # makes it so that the output won't be outside the maze
                    continue
                if get_value((position[0] + i, position[1] + j)) == -1:  # if the neighbor is a wall
                    continue
                if not (i == 0 or j == 0):  # non-resilient
                    continue

                output.append((position[0] + i, position[1] + j))

        return output

    def get_value(position):  # returns the value of the input position in the maze
        # non-resilient
        
        return maze[position[0]][position[1]]

    infinity = float('inf')  # useful for making future code more readable
    
    # list of Nodes that need to be explored
    to_do = [starting_position]
    
    # the complete history of the to_do list, not used in algorithm, just for visual output
    to_do_history = []

    # list of Nodes that have been explored
    explored = [starting_position]

    # the complete history of the explored list, not used in algorithm, just for visual output
    explored_history = []
    
    # list of the paths taken to explore nodes, not used in algorithm, just for visual output
    path_history = []
    
    # gScore[n] on wiklipedia will be shortest_path_cost[n] on here
    shortest_path_cost = {}
    # set all position's shortest path cost value to infinity
    for i in range(len(maze)):  # non-resilient
        for j in range(len(maze[i])):  # non-resilient
            shortest_path_cost[(i, j)] = infinity  # non-resilient
    shortest_path_cost[starting_position] = get_value(starting_position)  # the cost to reach the starting position will be 0
    
    # came_from[n] is the neighboring node directly preceding n on the shortest path to n
    came_from = {}
    
    # heuristic_dict[n] gives the output of heuristic(n) without having to re-run it
    # probably wouldn't be necessary in a maze like project euler problem 67
    # heuristic is used to prioritise movement towards the end position
    heuristic_dict = {}  # non-resilient
    for i in range(len(maze)):  # non-resilient
        for j in range(len(maze[i])):  # non-resilient
            heuristic_dict[(i, j)] = heuristic((i, j), goal_position)

    # For node n, to_finish_through[n] := gScore[n] + h(n).fScore[n] represents our current best guess as to
    # how short a path from start to finish can be if it goes through n.
    to_finish_through = {}
    # set all position's path length value to infinity
    for i in range(len(maze)):  # non-resilient
        for j in range(len(maze[i])):  # non-resilient
            to_finish_through[(i, j)] = infinity  # non-resilient
    
    
    def reconstruct_path(came_from: dict, current: tuple):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path

    while len(to_do) > 0:  # while there are still to_do nodes
        value_dict = {pos: maze[pos[0]][pos[1]] for pos in to_do}  # the values of each position on the to_do list
        current_node = min(to_do, key=heuristic_dict.get)  # get the position with the lowest heuristic value that's in the to_do list
        
        

        if current_node == goal_position:  # if the current node is the destination, end
            return Output(final_path=reconstruct_path(came_from=came_from, current=current_node), maze=maze, to_do_history=to_do_history, path_history=path_history, explored_history=explored_history)

        to_do.remove(current_node)  # node has been "explored", so remove it from the to_do list
        explored.append(current_node)
        
        to_do_history.append(list(to_do))  # add the current to_do list to the to_do_history list, the list function is run so that the element in to_do_history won't update when to_do updates
        path_history.append(reconstruct_path(came_from=came_from, current=current_node))  # adds the path taken to the current node to path_history
        explored_history.append(list(explored))

        for neighbor in neighbors(current_node):
            # tentative_shortest_path_cost is the distance from start to the neighbor through current
            
            tentative_shortest_path_cost = shortest_path_cost[current_node] + get_value(neighbor)
            if tentative_shortest_path_cost < shortest_path_cost[neighbor]:  # if the new path to neighbor is the shortest path to neighbor
                came_from[neighbor] = current_node

                shortest_path_cost[neighbor] = tentative_shortest_path_cost
                heuristic_dict[neighbor] = tentative_shortest_path_cost + heuristic(neighbor, goal_position)
                
                if neighbor not in to_do:
                    to_do.append(neighbor)
    
    
    return Output(failed=True)


def animate_path(show_final_path_frames: int, maze: list, path_color: list, explored_color: list, to_do_color: list, final_path: list, path_history: list, to_do_history: list, explored_history: list, fps: float, save_files=False, gif_frame_skip=1, frame_skip=0):
    if type(final_path) == str:  # if final_path is a string
        return  # that being a string indicates that the algorithm failed to find a path, exit the function
    
    if len(path_history) != len(to_do_history):  # if the lengths of ant of the history lists are different
        return  # something went wrong, just exit

    fig, ax = plt.subplots()  # initialising plot

    display_array = numpy.array(maze)

    graph = plt.imshow(display_array)  # add the array to the window


    # color the maze
    max_value_in_maze = max(unzip(maze))
    min_value_in_maze = min(unzip(maze))
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            current_value = maze[i][j]

            if maze[i][j] == -1:
                maze[i][j] = [0, 0, 0]
            else:
                maze[i][j] = (current_value - min_value_in_maze) / (max_value_in_maze - min_value_in_maze)

                maze[i][j] = int(maze[i][j] * 128 + 127)
                maze[i][j] = [maze[i][j], maze[i][j], maze[i][j]]

    def animate(i):
        i *= frame_skip
        
        display_array = numpy.array(maze)  # reset display_array
    
        if i >= len(path_history):  # if it's on the last frame of the animation
            # show the final path
            for node in final_path:  # for each node in the final path
                display_array[node] = path_color
            # hold the animation on the last frame so that the final path is visible
            pass
        
            graph.set_data(display_array)
        
            # save frame
            if save_files:
                plt.savefig(f"gif_folder/frame_{i}.jpg")
                if (i/frame_skip) == len(path_history) + show_final_path_frames - 1:  # if it's on the final frame of the animation
                    make_gif("gif_folder", fps=fps, num_frames=i + 1)
            return fig
    
        # pos = (1, 1)
        # display_array[pos] = display_array[pos] - 1
        for node in explored_history[i]:  # for each explored node on the frame 'i'
            display_array[node] = explored_color
        for node in to_do_history[i]:  # for each node in to_do on frame 'i'
            display_array[node] = to_do_color
        for node in path_history[i]:  # for each node in the path of frame 'i'
            display_array[node] = path_color
    
        graph.set_data(display_array)
    
        # save frame
        if save_files:
            plt.savefig(f"gif_folder/frame_{i}.jpg")
        
        
        graph.set_data(display_array)
        
        # save frame
        if save_files and not i % gif_frame_skip:
            plt.savefig(f"gif_folder/frame_{i//gif_frame_skip}.jpg")
        return fig
    

    ani = FuncAnimation(fig, animate, frames=len(path_history) + show_final_path_frames, interval=0)
    plt.show()


def calculate_path_cost(maze, path):
    
    # non-resilient
    total = 0
    for i in path[1:]:
        total += maze[i[0]][i[1]]
    return total


# arguments ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

maze_x = 100
maze_y = 100
maze_value_range = 10  # int showing the amount of different values that can appear on the maze
maze_wall_rate = 0.3

starting_position = (0, 0)
goal_position = (maze_y-1, maze_x-1)

animation_save_files = False
animation_show_final_path_frames = 120
animation_fps = 30
animation_frame_skip = 10
animation_gif_frame_skip = 30  # how many frames of the original animation per gif frame (used to save space)
animation_path_color = [238, 255, 13]
animation_explored_color = [85, 208, 230]
animation_to_do_color = [115, 227, 113]

# arguments ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


start = time()


maze = make_maze(x=maze_x, y=maze_y, value_range=maze_value_range, wall_rate=maze_wall_rate, starting_position=starting_position, ending_position=goal_position)


maze = manual_maze("""3389468957195191621612774424781291213569193831469897426487912119997336538328929529524567932999764397
9696723499379962193291388224211989319681442926919991941367126322911153337169884311818994691998142818
7814276813122519516197651882922789689697678423766816984171349893557699216293531566171577752862999189
8695248971983831118888345832382379869828991721417166494233961814479616298512617993916199484169768863
9438388913239281978754611917891972779819131371394848464949495975929194988978263297929499659581689931
1969369226881869218169885835118239749691599946731678781981449349938381248989441932791317925352759942
7123779172193954721688869139758196888794211367981479259955577415226913521927979417918717189391166677
7943535198721159911645992488162197692426999278117467587266221897325185828271274831492759999389981518
7887619766952832911156929833966381186297622871182811544331199411149635721326489819882214998153596213
4648859361999318828881375191941748791136972816261717334561674919882252411527611484116833231296411618
2212833947623292922961462362496994195995499987579663947111959949488113399114998685977193848697511116
3969643397751161263327478692813671171847775693111943664189164399861484521889181142821292229863551167
8151199761993969924113338516875823735898546129898975198189479571912373955912635949179871968717523778
1414763299794192255175491986733883171469919251932778658489558798293791213523573113651247139189979782
2914799981456921799818211127852862317673886288925766299289279774929166126931375969181699919238741667
2698477149739411976291599914938267421975388151967614241699299899988245295611437935247966657591969778
5111685483932883334989653569619111635366491297319621661611923917367917419122271997221566146467196419
1788715511737849818313857123597911255911246797868723244979271345629978187199964189721899894769193943
7799127514813556529152157494481412175528363357891151919575984268994117588197854536971298218293132975
5191589471484411568918992274115248889993179919351221221161883968825191864548331595791166197198127399
7134111721142222921564942388988611329462613295296451198969241921221266311131154727752149669939921451
9231215817921161996961419339655958616296399432964859979861791388899829432588951126486912217259167832
7288383496924661955553368918859194248378665392455721328875918986215811121839385268238274922337419599
5398269731871319317925166181127791822121922995312494296518745191129851711261599329258446769978513371
7519911347619499998183144921992923383195118218717115338898919755349399522578587858498623729969364439
7799129986113515119414994929919839999292571679437875189815112227229198341931879969713161977291229411
7935126516249816371388747122241374929559962158944822262859887411378978918828979468577636757199381622
9619932788677832767569586827293945491235991219829538299775962554614497461119146288916992936563148974
9274593916115941959559369931419911191379991777178297819173418914511525194974235391698919979979326488
4197882683459929888814912339873331135564287344534278415271819298251894178819155117432111915162524437
5433115493211699435793985889978218741372736831392184941927199689192129119118358892894452459744481878
1657317855321186979916212557229189126829384985198684586779197782862729537145651985526969534395252897
2116541971938549922767831913227758894366215369534934499842711566152199691337438913318633821118197342
1983831969661227223991853684189749239981961379472232986914971478185157749262379825981151116111391621
4151918985339445215718286389479225192892875919181211445498736766932899588886875199965186938911692151
9589959948482774911319126412426417169841631999115519761879839629477452625111738644112591572966125182
5966781638868275174869483219311675196979665689337619115451698221599569139959921748975692418154161215
9139294185336919134967481513876592918798682562619962843826889317923849891328947261161198818175918218
8953314967869319325225222819348241292514342588988219916851746196188279568772199878152119741943687547
3623975797337114736793127911219891616989941371351811389988939226193113299227559114939998824933596189
4792198337789991513998111115624229327919716399125683154671864613488996913763986297989392214918799549
9582156162119767191968293111982148921117852365798973396484976367431679823788317266841239191591867913
1832427961318389595862199619377811881597987152846538794932933914386173429144926381195992913531469974
9691161819686318892917368671921731596121389837296929711715188131389941519996979627999199591991447237
9215185378797971821395199789981278927982378673997259983291882657292328181898316839993294691643299918
5923989193525399996283385481919987227168623211393319139671359456396994159292245311434669927135312996
1884625283194198884964982369891828185782113217292219114831166257469135571961533825926199954629417349
6111949696979799811373595191234733819562148384923973895979211192836681941894691812541844218336511339
1249191116838794499929495172321833926959698393418491245194131441699723727197797164925165634819657988
2759999328431922998429144768748826516915128299397318379136812516257417734757929719859994463953714737
8412564991848221147123362451415229129914455129828797121813332839683412625979532529977754174985173312
9742465485298332327612945182849383772991183957251966333793827252746988179943996447941239888251319238
3671711995796174979139941113938431582899299195925181152951151397591597629215954959631917337368878879
8572911294992269273413272895293193181435721329699969666966951818961271149911584782639851838411128653
8639931138942796916933259811589581615765937985989129659479818479119983893281668137497597192317551992
6929874911682426233195283262476189294111737918136624982649316917143913141994718522713155795784823193
2145599768916692449167618961187975991119259336856741211277385399699249992884451697138386924211373474
1873372796987436943131312835256225161714919262191652723119832296197636121922849295821535283538916762
2265491922334921624297523951291195949212943689816426171136931252288434457431397165991971931349772995
7948989914979949319739929811162848197242594572224894223185178399413154839347668173522499315412849841
6512298397749956981193929432127349154191232992723791712799399549897128999753472377332829713939844361
8318256681893818447127522891199585141899915919979948991198619491619713884191379396176914496755588291
9719173813321238564133749261368914593111277529319912988479688926274151821671943875362235784319616865
7311898511997891581895653229799294836812924356416919365514454829976974192691985278349914191448329111
8817168994132999198368881732584351379817759819491141482294889256135317795783961892249539482715816792
5896213329918833292293911783651125119972217853144928455791518991884939185238272581218289359229314319
3617297154332885617165178984393881998243916999482731981356353638517787187224712118912138456814919133
3535392788469846427491259255498814338119727966121371696135591892721238769329394666922458135939662918
8978713293695632285874319439519191528111993197233921631728174431879983883769192996395656124959349113
3992992899774912938872961918235813659794322493593751297111889487248153992595796911879156514968951999
1471382318485251196198453195432318184626959713594972656999554854114159575689998859276493989363544466
1914898952712968817183517192313388723953142976199329737616825511227233527517211968929219797929437618
1272193394839879255172979571591474229318787739666234782976964698287147711148116199239659167242671661
3654331179111349792874431999652732599285596829861644244916181997299656916269261632958231258617213317
5271993216416317872662593141986931579961338264743488457429991961919194199494921524355223119718991439
8891127916699131371791479939236581692148538871244496989159269311413633195198718812918128848518994988
9813245138697623492999496366863719931948111957276344647113779158171811823828433971481294592115619244
7516259831363511251771637699119734921996998577511252532589636172425851888377191246923711967978189775
6111913728638886247111116979292975943492991527751161823641178386991291519931894872596111234978997189
6717611188354611555241979133156559369829996819272871115282379926296281342657121174182835474195629231
9878715996669514192488194781194942751813814811131235169867122578931899198195111198493947432296924683
6438791983961828369724259934167221814427296949178238944938113283325719271944992987711875824461298118
5195451989571871525468815787519187119118618952998991277251131269781141749196471955457995478697886443
2959822826382133121882563591589713899811521478342383738916889783729421618338343491998126218314915171
2631923589942121913799763789216419692361914546181659457617782336821942122188811851383138241753649489
9992289429129341984433974315622294639852937888122388158231848984421542369311819216942259519936215481
3349464198938889131791931113242889116216542154192949399985916979215922819649429733791931278834994712
2711163996265866311352189994619383246984822916993759144813151815299488934194181213986395739964258137
2174745854781389832416122151952317564328122421644357343168175979482567885291413293993996938675362939
7579317491998958176743892229898112525411948981982915814899523891116829898692464736273842737917296699
3943981243848348897569424647878277184921348331489927195159961856893716141223583261711275595691219931
3894196153581992667198693188995168422145894399925925177632929527319482971447681924549328174989362194
1938318592297692293779317816386969248919752566557915139919614121911113231921769686888496487914655978
6895761659942389992173411367761419996886179248923636115218383311645919491759834821511123413585319447
1714992394989415348976956296915691959991463176292895154115458928981111111658281985886256787111317281
6111427283128793746188988196974159169972211989199319534919763791982577916819976818935222459811561994
9112369767928176941992947719126492789972131111267191921699379681595192912698118983672353699932949194
3691316978229429439969257627496969959919188996129916822468586994891228183171556583538735756981191184
1889273619911397689813219925751671129554482532979374999962138151188171289238825995754135581779326797
2865692499363943591911977961231313686973849878387599963859497937346925669637729745891898997393123146""")

with open('readme.txt', 'w') as f:
    for i in maze:
        for j in i:
            f.write(str(j) + " ")
        f.write("end\n")

output = a_star(maze=maze, starting_position=starting_position, goal_position=goal_position)  # non-resilient

elapsed = time() - start



if not output.failed:
    print(f"elapsed: {elapsed}s")

    print(f'cost: {calculate_path_cost(maze, output.final_path)}')
    
    animate_path(maze=output.maze, path_color=animation_path_color, explored_color=animation_explored_color, to_do_color=animation_to_do_color, final_path=output.final_path, path_history=output.path_history, to_do_history=output.to_do_history, explored_history=output.explored_history, save_files=animation_save_files, show_final_path_frames=animation_show_final_path_frames, fps=animation_fps, gif_frame_skip=animation_gif_frame_skip, frame_skip=animation_frame_skip)
else:
    print("Pathfinding failed :(")
