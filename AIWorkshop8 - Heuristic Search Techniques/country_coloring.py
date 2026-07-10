from simpleai.search import CspProblem, backtrack

def constraint_adjacent(regions, colors):
    """
    Constraint function: Two adjacent regions must have different colors.
    Args:
        regions: Tuple of two adjacent region names (e.g., ('Beijing', 'Hebei')).
        colors: Tuple of colors assigned to the two regions (e.g., ('red', 'green')).
    Returns:
        bool: True if colors are different (constraint satisfied), False otherwise.
    """
    return colors[0] != colors[1]

def find_min_coloring(all_regions, adjacency_pairs):
    """
    Test 2→3→4 colors to find the minimum number needed for coloring.
    Args:
        all_regions: List of all 34 Chinese provincial-level regions.
        adjacency_pairs: List of tuples, each representing a pair of adjacent regions.
    Returns:
        tuple: (min_color_count, color_solution) — minimum colors and region-color mapping.
    """
    # Color sets for testing (ordered by increasing count)
    color_options = {
        2: ['red', 'green'],               # Test 2 colors first
        3: ['red', 'green', 'blue'],       # Test 3 if 2 fails
        4: ['red', 'green', 'blue', 'gray']# Test 4 if 3 fails (theorem guarantee)
    }

    # Iterate from 2 to 4 colors to find the minimum
    for color_count in [2, 3, 4]:
        # Define domain: each region can use any color in the current color set
        domains = {region: color_options[color_count] for region in all_regions}
        
        # Define constraints: all adjacent pairs must satisfy the color difference rule
        constraints = [(pair, constraint_adjacent) for pair in adjacency_pairs]
        
        # Initialize CSP problem and solve with backtracking (efficient for finite domains)
        problem = CspProblem(all_regions, domains, constraints)
        solution = backtrack(problem)  # Returns None if no valid solution exists
        
        # If solution found, return immediately (since we test in ascending order, it's optimal)
        if solution:
            return (color_count, solution)
    
    # Fallback (four-color theorem ensures 4 colors work, so this line is rarely reached)
    return (4, None)

if __name__ == '__main__':
    # --------------------------
    # Step 1: Define all 34 provincial-level regions of China
    # --------------------------
    all_china_regions = [
        # Municipalities (4)
        'Beijing', 'Tianjin', 'Shanghai', 'Chongqing',
        # Provinces (23)
        'Hebei', 'Shanxi', 'Liaoning', 'Jilin', 'Heilongjiang',
        'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi',
        'Shandong', 'Henan', 'Hubei', 'Hunan', 'Guangdong',
        'Hainan', 'Sichuan', 'Guizhou', 'Yunnan', 'Shaanxi',
        'Gansu', 'Qinghai', 'Taiwan',
        # Autonomous Regions (5)
        'InnerMongolia', 'Guangxi', 'Tibet', 'Ningxia', 'Xinjiang',
        # Special Administrative Regions (2)
        'HongKong', 'Macau'
    ]

    # --------------------------
    # Step 2: Define actual adjacency pairs (geographic land borders)
    # Note: Adjacency is verified based on China's administrative geography; no maritime borders.
    # --------------------------
    adjacency_pairs = [
        # Beijing's adjacents
        ('Beijing', 'Hebei'), ('Beijing', 'Tianjin'),
        # Tianjin's adjacents
        ('Tianjin', 'Hebei'), ('Tianjin', 'Beijing'),
        # Shanghai's adjacents
        ('Shanghai', 'Jiangsu'), ('Shanghai', 'Zhejiang'),
        # Chongqing's adjacents
        ('Chongqing', 'Sichuan'), ('Chongqing', 'Guizhou'), ('Chongqing', 'Hubei'), ('Chongqing', 'Hunan'),
        # Hebei's adjacents
        ('Hebei', 'Beijing'), ('Hebei', 'Tianjin'), ('Hebei', 'Shanxi'), ('Hebei', 'Liaoning'), 
        ('Hebei', 'Shandong'), ('Hebei', 'Henan'), ('Hebei', 'InnerMongolia'),
        # Shanxi's adjacents
        ('Shanxi', 'Hebei'), ('Shanxi', 'Shaanxi'), ('Shanxi', 'Henan'), ('Shanxi', 'InnerMongolia'),
        # Liaoning's adjacents
        ('Liaoning', 'Hebei'), ('Liaoning', 'Jilin'), ('Liaoning', 'InnerMongolia'),
        # Jilin's adjacents
        ('Jilin', 'Liaoning'), ('Jilin', 'Heilongjiang'), ('Jilin', 'InnerMongolia'),
        # Heilongjiang's adjacents
        ('Heilongjiang', 'Jilin'), ('Heilongjiang', 'InnerMongolia'),
        # Jiangsu's adjacents
        ('Jiangsu', 'Shanghai'), ('Jiangsu', 'Zhejiang'), ('Jiangsu', 'Anhui'), ('Jiangsu', 'Shandong'),
        # Zhejiang's adjacents
        ('Zhejiang', 'Shanghai'), ('Zhejiang', 'Jiangsu'), ('Zhejiang', 'Anhui'), ('Zhejiang', 'Fujian'),
        # Anhui's adjacents
        ('Anhui', 'Jiangsu'), ('Anhui', 'Zhejiang'), ('Anhui', 'Fujian'), ('Anhui', 'Jiangxi'), 
        ('Anhui', 'Henan'), ('Anhui', 'Shandong'),
        # Fujian's adjacents
        ('Fujian', 'Zhejiang'), ('Fujian', 'Anhui'), ('Fujian', 'Jiangxi'), ('Fujian', 'Guangdong'), ('Fujian', 'Taiwan'),
        # Jiangxi's adjacents
        ('Jiangxi', 'Anhui'), ('Jiangxi', 'Fujian'), ('Jiangxi', 'Guangdong'), ('Jiangxi', 'Hunan'), ('Jiangxi', 'Hubei'),
        # Shandong's adjacents
        ('Shandong', 'Hebei'), ('Shandong', 'Jiangsu'), ('Shandong', 'Anhui'), ('Shandong', 'Henan'),
        # Henan's adjacents
        ('Henan', 'Hebei'), ('Henan', 'Shanxi'), ('Henan', 'Shaanxi'), ('Henan', 'Hubei'), 
        ('Henan', 'Anhui'), ('Henan', 'Shandong'),
        # Hubei's adjacents
        ('Hubei', 'Henan'), ('Hubei', 'Jiangxi'), ('Hubei', 'Hunan'), ('Hubei', 'Chongqing'), 
        ('Hubei', 'Shaanxi'), ('Hubei', 'Anhui'),
        # Hunan's adjacents
        ('Hunan', 'Hubei'), ('Hunan', 'Jiangxi'), ('Hunan', 'Guangdong'), ('Hunan', 'Guangxi'), 
        ('Hunan', 'Guizhou'), ('Hunan', 'Chongqing'),
        # Guangdong's adjacents
        ('Guangdong', 'Fujian'), ('Guangdong', 'Jiangxi'), ('Guangdong', 'Hunan'), ('Guangdong', 'Guangxi'), 
        ('Guangdong', 'HongKong'), ('Guangdong', 'Macau'), ('Guangdong', 'Hainan'),
        # Hainan's adjacents (only adjacent to Guangdong via land? No—Hainan is an island; adjust if considering maritime, but here we use land borders)
        ('Hainan', 'Guangdong'),  # Treat as adjacent for practical coloring (adjust if needed)
        # Sichuan's adjacents
        ('Sichuan', 'Chongqing'), ('Sichuan', 'Guizhou'), ('Sichuan', 'Yunnan'), ('Sichuan', 'Shaanxi'), 
        ('Sichuan', 'Gansu'), ('Sichuan', 'Qinghai'), ('Sichuan', 'Tibet'),
        # Guizhou's adjacents
        ('Guizhou', 'Chongqing'), ('Guizhou', 'Sichuan'), ('Guizhou', 'Yunnan'), ('Guizhou', 'Hunan'), ('Guizhou', 'Guangxi'),
        # Yunnan's adjacents
        ('Yunnan', 'Sichuan'), ('Yunnan', 'Guizhou'), ('Yunnan', 'Guangxi'), ('Yunnan', 'Tibet'),
        # Shaanxi's adjacents
        ('Shaanxi', 'Shanxi'), ('Shaanxi', 'Henan'), ('Shaanxi', 'Hubei'), ('Shaanxi', 'Chongqing'), 
        ('Shaanxi', 'Sichuan'), ('Shaanxi', 'Gansu'), ('Shaanxi', 'Ningxia'), ('Shaanxi', 'InnerMongolia'),
        # Gansu's adjacents
        ('Gansu', 'Shaanxi'), ('Gansu', 'Sichuan'), ('Gansu', 'Qinghai'), ('Gansu', 'Ningxia'), 
        ('Gansu', 'Xinjiang'), ('Gansu', 'InnerMongolia'),
        # Qinghai's adjacents
        ('Qinghai', 'Sichuan'), ('Qinghai', 'Gansu'), ('Qinghai', 'Tibet'), ('Qinghai', 'Xinjiang'), ('Qinghai', 'Ningxia'),
        # Taiwan's adjacents
        ('Taiwan', 'Fujian'),  # Adjacent via strait (practical for coloring)
        # InnerMongolia's adjacents
        ('InnerMongolia', 'Hebei'), ('InnerMongolia', 'Shanxi'), ('InnerMongolia', 'Liaoning'), 
        ('InnerMongolia', 'Jilin'), ('InnerMongolia', 'Heilongjiang'), ('InnerMongolia', 'Shaanxi'), 
        ('InnerMongolia', 'Gansu'), ('InnerMongolia', 'Ningxia'),
        # Guangxi's adjacents
        ('Guangxi', 'Hunan'), ('Guangxi', 'Guangdong'), ('Guangxi', 'Guizhou'), ('Guangxi', 'Yunnan'),
        # Tibet's adjacents
        ('Tibet', 'Sichuan'), ('Tibet', 'Yunnan'), ('Tibet', 'Qinghai'), ('Tibet', 'Xinjiang'),
        # Ningxia's adjacents
        ('Ningxia', 'Shaanxi'), ('Ningxia', 'Gansu'), ('Ningxia', 'Qinghai'), ('Ningxia', 'InnerMongolia'),
        # Xinjiang's adjacents
        ('Xinjiang', 'Gansu'), ('Xinjiang', 'Qinghai'), ('Xinjiang', 'Tibet'),
        # HongKong's adjacents
        ('HongKong', 'Guangdong'),
        # Macau's adjacents
        ('Macau', 'Guangdong')
    ]

    # --------------------------
    # Step 3: Find minimum colors and solve coloring
    # --------------------------
    min_colors, color_solution = find_min_coloring(all_china_regions, adjacency_pairs)

    # --------------------------
    # Step 4: Print results (sorted by region type for readability)
    # --------------------------
    print("=" * 60)
    print(f"Minimum number of colors needed for all Chinese provinces: {min_colors}")
    print("\nRegion → Color Mapping (no adjacent regions share the same color):")
    print("-" * 60)

    # Categorize regions for organized output
    municipalities = ['Beijing', 'Tianjin', 'Shanghai', 'Chongqing']
    provinces = ['Hebei', 'Shanxi', 'Liaoning', 'Jilin', 'Heilongjiang',
                 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi',
                 'Shandong', 'Henan', 'Hubei', 'Hunan', 'Guangdong',
                 'Hainan', 'Sichuan', 'Guizhou', 'Yunnan', 'Shaanxi',
                 'Gansu', 'Qinghai', 'Taiwan']
    autonomous_regions = ['InnerMongolia', 'Guangxi', 'Tibet', 'Ningxia', 'Xinjiang']
    sar = ['HongKong', 'Macau']

    # Print each category
    print("\n1. Municipalities:")
    for reg in sorted(municipalities):
        print(f"   {reg:<12} → {color_solution[reg]}")
    
    print("\n2. Provinces:")
    for reg in sorted(provinces):
        print(f"   {reg:<12} → {color_solution[reg]}")
    
    print("\n3. Autonomous Regions:")
    for reg in sorted(autonomous_regions):
        print(f"   {reg:<12} → {color_solution[reg]}")
    
    print("\n4. Special Administrative Regions (SAR):")
    for reg in sorted(sar):
        print(f"   {reg:<12} → {color_solution[reg]}")
    
    print("=" * 60)