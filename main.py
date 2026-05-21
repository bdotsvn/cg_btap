import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import os
import sys
import tkinter as tk
from tkinter import filedialog

def trace_skeleton_dfs(skeleton):
    """
    Traces the skeleton using Depth First Search (DFS).
    This creates longer, more continuous strokes and perfectly closes gaps at intersections.
    """
    ys, xs = np.where(skeleton)
    skel_pixels = set(zip(xs, ys))
    
    # Precompute neighbors
    graph = {}
    for x, y in skel_pixels:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                if (x+dx, y+dy) in skel_pixels:
                    neighbors.append((x+dx, y+dy))
        graph[(x,y)] = neighbors
        
    unvisited = set(skel_pixels)
    curves = []
    
    # Prefer starting from endpoints
    endpoints = [node for node in skel_pixels if len(graph[node]) == 1]
    
    while unvisited:
        start_node = None
        for ep in endpoints:
            if ep in unvisited:
                start_node = ep
                break
        if not start_node:
            start_node = next(iter(unvisited))
            
        curve = [start_node]
        unvisited.remove(start_node)
        
        # Prepend a visited neighbor to close gaps when starting a new branch from a junction
        visited_neighbors = [n for n in graph[start_node] if n not in unvisited]
        if visited_neighbors:
            curve.insert(0, visited_neighbors[0])
            
        curr = start_node
        while True:
            unvisited_neighbors = [n for n in graph[curr] if n in unvisited]
            
            if not unvisited_neighbors:
                # Dead end. Append a visited neighbor to close the gap at the end
                if len(curve) > 1:
                    prev = tuple(curve[-2])
                    other_visited = [n for n in graph[curr] if n not in unvisited and n != prev]
                    if other_visited:
                        curve.append(other_visited[0])
                break
                
            next_node = unvisited_neighbors[0]
            curve.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
            
        if len(curve) >= 4:
            curves.append(np.array(curve))
            
    return curves

def image_to_diempixel(image_path, output_filename="diempixel.dat"):
    """
    1) Đọc ảnh, tìm xương chữ ký và xuất tọa độ pixel ra file diempixel.dat
    """
    print(f"Loading {image_path}...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return None, 0
        
    height, width = img.shape
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary // 255
    print("Skeletonizing...")
    skeleton = skeletonize(binary)
    
    print("Tracing paths using DFS...")
    segments = trace_skeleton_dfs(skeleton)
    
    with open(output_filename, 'w') as f:
        for i, segment in enumerate(segments):
            f.write(f"STROKE {i}\n")
            for pt in segment:
                f.write(f"{pt[0]} {pt[1]}\n")
                
    print(f"Exported pixel data to {output_filename}")
    return img, height

def read_diempixel(filename="diempixel.dat"):
    """
    1) Đọc dữ liệu tọa độ điểm từ file diempixel.dat
    """
    segments = []
    current_segment = []
    
    if not os.path.exists(filename):
        return []
        
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("STROKE"):
                if current_segment:
                    segments.append(np.array(current_segment))
                    current_segment = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    current_segment.append([float(parts[0]), float(parts[1])])
                    
    if current_segment:
        segments.append(np.array(current_segment))
        
    return segments

def Nip(u, knots, degree):
    """
    2) Tính toán các hàm basic Nip(u) của đường cong B-spline
    """
    n_knots = len(knots)
    n_control = n_knots - degree - 1
    N = np.zeros((len(u), n_control))
    for i in range(n_control):
        coeffs = np.zeros(n_control)
        coeffs[i] = 1.0
        spl = BSpline(knots, coeffs, degree)
        N[:, i] = spl(u)
    return N

def LSTBSplineRecontruction(points, n_control=20, degree=3):
    """
    3) Hàm tái tạo least-square approximation LSTBSplineRecontruction (Non-uniform)
    """
    x = points[:, 0]
    y = points[:, 1]
    m = len(points)
    
    if m < 4: return None
    
    # 1. Parameterization (Chord Length method)
    u = np.zeros(m)
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    u[1:] = np.cumsum(dists)
    if u[-1] > 0:
        u /= u[-1]
    else:
        u = np.linspace(0, 1, m)
        
    # 2. Generate NON-UNIFORM Knots
    n_knots = n_control + degree + 1
    knots = np.zeros(n_knots)
    
    # Clamped ends
    knots[:degree+1] = 0.0
    knots[-degree-1:] = 1.0
    
    # Internal non-uniform knots based on proper parameter sampling
    # We need (n_control - degree) internal knots. We sample them evenly from the non-uniform u array.
    if n_control > degree:
        indices = np.linspace(0, m - 1, n_control - degree + 2, dtype=int)[1:-1]
        for j, idx in enumerate(indices):
            knots[j + degree + 1] = u[idx]
    
    # 3. Build Basis Matrix N (m x n_control)
    N = Nip(u, knots, degree)
    
    # 4. Solve the Least Squares system: (N^T * N) * P = N^T * Q
    try:
        cp_x, _, _, _ = np.linalg.lstsq(N, x, rcond=None)
        cp_y, _, _, _ = np.linalg.lstsq(N, y, rcond=None)
        return knots, cp_x, cp_y, degree
    except Exception as e:
        print(f"Error in Least-Squares: {e}")
        return None

def export_bsplinecurve(filename, curves, img_height):
    """
    4) Xuất dữ liệu của đường cong tái tạo ra file dữ liệu bsplinecurve.dat theo chuẩn chương trình DUTMod/DISCO
    """
    if not curves:
        print("No curves to export.")
        return
        
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (knots, cp_x, cp_y, degree) in enumerate(curves):
            num_cp = len(cp_x)
            knot_type = 1 
            
            f.write("=============\n")
            f.write("[BSPLINECURVE]\n\n")
            f.write(f"{num_cp}, {degree}, {knot_type} // UNum, UDegree, UKnotType\n\n")
            f.write("// Control Points\n")
            
            for x, y in zip(cp_x, cp_y):
                # Lật tọa độ Y để hiển thị đúng hệ tọa độ Cartesian trong app DISCO
                y_flipped = img_height - y
                f.write(f"{x:.8f} {y_flipped:.8f} 0.00000000 1.00000000 0\n")
                
            f.write("\nUKnot\n")
            for k in knots:
                f.write(f"{k:.8f}\n")
                
            if i < len(curves) - 1:
                f.write("\n") 
                
    print(f"Exported {len(curves)} B-Spline curves to {filename}")

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chon anh chu ky",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
    )
    return file_path

if __name__ == "__main__":
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Vui long chon anh chu ky tu cua so (co the bi an phia sau)...")
        image_path = select_image_file()
        
    if not image_path:
        print("No image selected. Exiting.")
        sys.exit(0)
        
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    # BƯỚC 1: Xử lý ảnh và xuất file diempixel.dat
    img, img_height = image_to_diempixel(image_path, "diempixel.dat")
    
    if img is None:
        sys.exit(1)

    # BƯỚC 2: Đọc file diempixel.dat
    segments = read_diempixel("diempixel.dat")
    print(f"Read {len(segments)} segments from diempixel.dat")
    
    # BƯỚC 3: Xấp xỉ Least-Squares và gom data
    curves = []
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap='gray', alpha=0.3)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']
    
    for i, ordered_points in enumerate(segments):
        m = len(ordered_points)
        # Động học số điểm điều khiển: 1 điểm điều khiển cho mỗi 10 pixel
        n_cp = max(4, m // 10)
        
        # Nếu đoạn nét vẽ cực ngắn thì không cần dùng quá nhiều
        n_cp = min(n_cp, m)
        
        if n_cp < 4: continue
        
        result = LSTBSplineRecontruction(ordered_points, n_control=n_cp, degree=3)
        if result:
            knots, cp_x, cp_y, degree = result
            curves.append((knots, cp_x, cp_y, degree))
            
            spl = BSpline(knots, np.column_stack((cp_x, cp_y)), degree)
            u_fine = np.linspace(0, 1, 500)
            curve_pts = spl(u_fine)
            
            color = colors[i % len(colors)]
            plt.plot(curve_pts[:, 0], curve_pts[:, 1], color=color, linestyle='-', linewidth=2)
            
    plt.title(f'LST B-Spline Reconstruction ({len(curves)} strokes)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig('result.png')
    print("Saved visualization to result.png")

    # BƯỚC 4: Xuất file bsplinecurve.dat chuẩn DISCO
    if curves:
        export_bsplinecurve("bsplinecurve.dat", curves, img_height)
        print("\nHoan tat! Hay kiem tra 3 file: diempixel.dat, bsplinecurve.dat, result.png")
        try:
            os.startfile("result.png")
        except:
            pass
