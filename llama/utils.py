
def candidate_path(root):
    
    all_entities = []
    all_scores = []
    all_paths = []
    all_ids = {}   

    with open(root, 'r') as file:
        lines = file.readlines()

    i =0
    for line in lines:     
        line = line.strip()  
        parts = line.split('\t')  

        entities = []
        scores = []
        paths = []

        qid = int(parts[0])
        parts = parts[1:]
        for part in parts:
            if part == '':
                print('null', i)
                continue
            try:
                entity, score, path = part.split('|')
            except: 
                print(part)
                exit(0)
            entities.append(entity)
            scores.append(float(score)) 
            path = path.split(';')
            split_path = []
            for p in path:
                if 'self_loop' not in p and p != '' and p not in split_path:
                    split_path.append(p)
            split_path = ', '.join(split_path)
            paths.append(split_path)

        
        all_entities.append(entities)
        all_scores.append(scores)
        all_paths.append(paths)
        all_ids[qid] = i
        i += 1

    return all_entities, all_scores, all_paths, all_ids

