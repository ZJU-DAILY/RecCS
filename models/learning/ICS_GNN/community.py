import networkx as nx



class LocalCommunity(object):

    def __init__(self, args, upgraph):
        self.args = args
        self.upgraph = upgraph
    def evaluate_community(self, top_index, name,using_time):
        '''
        Evaluate the quality of the search community
        '''
        y_pred = [0] * len(top_index)
        y_true = [0] * len(top_index)
        ok = 0
        pos = 0
        target = self.upgraph.sg_targets[0].cpu().detach().numpy().tolist()
        for i in range(len(top_index)):
            y_pred[i] = self.upgraph.sg_predLabels[top_index[i]]
            y_true[i] = target[top_index[i]][0]
            if top_index[i] in self.upgraph.sg_posNodes or top_index[i] in self.upgraph.posforrank:
                pos+=1
            if(y_true[i]==1):
                ok+=1
        results= []
        pre=0
        results.append(ok/len(top_index))
        if (len(top_index) != pos):
            pre=(ok-pos)/(len(top_index)-pos)
        results.append(pre)
        if name not in self.upgraph.methods:
            self.upgraph.methods[name]= results
            self.upgraph.time_map[name] =  using_time
            self.upgraph.cntmap[name]=1
        else:
            self.upgraph.methods[name] =[self.upgraph.methods[name][i]+results[i] for i in range(len(results))]
            self.upgraph.time_map[name] += using_time
            self.upgraph.cntmap[name]+=1
        print(name + " Precision={:.4f}  Precision without posnode={:.4f} using {:.4f}s".format(results[0],results[1],using_time))
        return results[0]


    def locate_community_BFS(self, seed):
        '''
        Search community using bfs with swap
        '''
        cnodes = []
        cnodes.append(seed)
        pos =0
        while pos < len(cnodes) and pos < self.upgraph.TOPK_SIZE and len(cnodes) < self.upgraph.TOPK_SIZE:
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                if nb not in cnodes and len(cnodes) < self.upgraph.TOPK_SIZE:
                    cnodes.append(nb)
            pos = pos + 1
        for pos in range(len(cnodes)):
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                 pos1= pos+1
                 while pos1<len(cnodes) and nb not in cnodes:
                    next = cnodes[pos1]

                    if self.upgraph.sg_predProbs[self.upgraph.mapper[nb]]>self.upgraph.sg_predProbs[self.upgraph.mapper[next]]:
                        cnodes[pos1] = nb
                    pos1 = pos1 +1
        topk = [node for node in cnodes] # id in the whole graph
        return topk


    def locate_community_greedy_graph_prepath(self, seed):
        '''
        Search community using  greedy-G
        '''
        cnodes = []
        cnodes.append(seed)
        pos = 0
        while pos < len(cnodes):
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                if nb not in cnodes:
                    cnodes.append(nb)
            pos = pos + 1
        topkidx = []
        topkidx.append(seed)
        for iter in range(self.upgraph.TOPK_SIZE):
            if (len(topkidx) == self.upgraph.TOPK_SIZE): break
            candidates = [-1.0] * 3 * self.upgraph.TOPK_SIZE
            for num in range(len(topkidx)):
                candidates[num] = 1.0
            probs = [-1.0] * len(self.upgraph.subgraph.nodes)
            hops = [1] * len(self.upgraph.subgraph.nodes)
            paths = self.get_all_path(self.upgraph.subgraph, topkidx,cnodes)
            for i in range(len(cnodes)):
                cnode = cnodes[i]
                if cnode in topkidx:
                    continue
                prob = self.upgraph.sg_predProbs[self.upgraph.mapper[cnode]]
                if prob < min(candidates):
                    continue
                for x in paths[i]:
                    prob = prob + self.upgraph.sg_predProbs[self.upgraph.mapper[x]]
                probs[i] = prob
                hops[i] = len(paths[i]) + 1
                prob = prob / hops[i]
                idx = candidates.index(min(candidates))
                candidates[idx] = prob
            for i in range(len(probs)):
                probs[i] = probs[i] / hops[i]
            maxValueIdx = probs.index(max(probs))
            if len(topkidx) < self.upgraph.TOPK_SIZE:
                topkidx.append(cnodes[maxValueIdx])
            for x in paths[maxValueIdx]:
                if len(topkidx) < self.upgraph.TOPK_SIZE:
                    topkidx.append(x)
        topk = [idx for idx in topkidx] # id in the whole graph
        return topk

    def get_all_path(self, pgraph, topkidx,cnodes):
        '''
        Get the community to the other nodes' shortest paths
        '''
        g = nx.Graph()
        seed = topkidx[0]
        for u, v in pgraph.edges:
            if (u in topkidx) and (v in topkidx):
                continue
            else:
                if (u in topkidx): u = seed
                if (v in topkidx): v = seed
                g.add_edge(u, v)
        p1 = nx.shortest_path(g, source=seed)
        paths = [[]] * len(pgraph.nodes)
        for item in p1.keys():
            idx=cnodes.index(item)
            path = p1[item][1:]
            path.reverse()
            paths[idx] = path[1:]
        return paths
