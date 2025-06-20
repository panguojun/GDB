import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Union, Optional, Tuple
import re
import pickle
import os
from pathlib import Path

# Define geometry shape types
Point = Dict[str, float]
Line = Dict[str, Union[Point, str]]
Rectangle = Dict[str, Union[Point, float, str]]
Circle = Dict[str, Union[Point, float, str]]
MeshTriangle = Dict[str, List[Point]]
Geometry = Union[Point, Line, Rectangle, Circle, MeshTriangle]

class GDB:
    def __init__(self, cache_file: str = 'gdb_cache.pkl'):
        """Initialize geometry database with optional disk cache"""
        self.cache_file = cache_file
        self.geometries: List[Geometry] = []
        self.next_id = 1
        
        # Try to load from cache if file exists
        self.load_from_cache()
    
    def load_from_cache(self):
        """Load database from cache file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.geometries = data.get('geometries', [])
                    self.next_id = data.get('next_id', 1)
                    print(f"Loaded database from cache with {len(self.geometries)} geometries")
            except Exception as e:
                print(f"Warning: Failed to load cache file - {e}")
    
    def save_to_cache(self):
        """Save current database state to cache file"""
        try:
            data = {
                'geometries': self.geometries,
                'next_id': self.next_id
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache file - {e}")
    
    def add_geometry(self, geom_type: str, **kwargs) -> int:
        """
        Add a geometry shape to the database
        :param geom_type: Geometry type ('point', 'line', 'rectangle', 'circle', 'mesh_triangle')
        :param kwargs: Geometry properties
        :return: Assigned ID
        """
        geom = {'type': geom_type, 'id': self.next_id, **kwargs}
        self.geometries.append(geom)
        self.next_id += 1
        self.save_to_cache()
        return geom['id']
    
    def read_obj_file(self, file_path: str):
        """
        Read an OBJ file and add triangles to the geometry database
        :param file_path: Path to the OBJ file
        """
        vertices = []

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if parts[0] == 'v':  # Vertex
                        vertex = {'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3]) if len(parts) > 3 else 0.0}
                        vertices.append(vertex)
                    elif parts[0] == 'f':  # Face
                        face_indices = []
                        for part in parts[1:]:
                            # Handle different OBJ face formats (v/vt/vn)
                            vertex_part = part.split('/')[0]
                            if vertex_part:
                                face_indices.append(int(vertex_part) - 1)
                        
                        if len(face_indices) >= 3:  # Handle both triangles and polygons
                            # Triangulate polygon if necessary
                            for i in range(1, len(face_indices) - 1):
                                triangle = [
                                    vertices[face_indices[0]],
                                    vertices[face_indices[i]],
                                    vertices[face_indices[i+1]]
                                ]
                                self.add_geometry('mesh_triangle', vertices=triangle)
            print(f"Successfully imported OBJ file: {file_path}")
            self.save_to_cache()
        except Exception as e:
            print(f"Error reading OBJ file: {e}")
    
    def execute_sql(self, sql: str) -> List[Geometry]:
        """
        Execute SQL query (simplified, does not use sqlparse)
        Supports format: SELECT [*|attribute list] FROM geometries [WHERE conditions]
        :param sql: SQL query string
        :return: Matching geometry shapes list
        """
        # Convert to lowercase and remove extra spaces
        sql = ' '.join(sql.lower().split())
        
        # Validate query format
        if not sql.startswith('select '):
            raise ValueError("SQL must start with SELECT")
        
        from_idx = sql.find(' from ')
        if from_idx == -1:
            raise ValueError("SQL must contain FROM clause")
        
        select_part = sql[7:from_idx].strip()
        remainder = sql[from_idx + 6:].strip()
        
        # Parse table name and conditions
        table_end = remainder.find(' where ')
        if table_end == -1:
            table_name = remainder
            where_part = ''
        else:
            table_name = remainder[:table_end].strip()
            where_part = remainder[table_end + 7:].strip()
        
        if table_name != 'geometries':
            raise ValueError("Only 'geometries' table is supported")
        
        # Parse selected fields
        if select_part == '*':
            select_fields = ['*']
        else:
            select_fields = [f.strip() for f in select_part.split(',')]
        
        # Parse WHERE conditions
        conditions = []
        if where_part:
            and_conditions = [c.strip() for c in where_part.split(' and ')]
            for cond in and_conditions:
                match = re.match(r'([a-z_]+)\s*(=|!=|>|<|>=|<=)\s*(.+)', cond)
                if not match:
                    raise ValueError(f"Invalid condition: {cond}")
                field, operator, value = match.groups()
                
                # Handle special cases for mesh triangles
                if field == 'vertices':
                    raise ValueError("Direct vertex comparison not supported, use type='mesh_triangle'")
                
                if (value.startswith("'") and value.endswith("'")) or \
                   (value.startswith('"') and value.endswith('"')):
                    value = value[1:-1]
                else:
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass  # Keep as string
                conditions.append((field, operator, value))
        
        # Execute query
        results = []
        for geom in self.geometries:
            if self._matches_conditions(geom, conditions):
                if '*' in select_fields:
                    results.append(geom)
                else:
                    filtered_geom = {'id': geom['id'], 'type': geom['type']}
                    for field in select_fields:
                        if field in geom:
                            filtered_geom[field] = geom[field]
                        elif field == 'vertices' and geom['type'] == 'mesh_triangle':
                            filtered_geom['vertices'] = geom['vertices']
                    results.append(filtered_geom)
        
        return results
    
    def _matches_conditions(self, geom: Geometry, conditions: List[Tuple[str, str, object]]) -> bool:
        """Check if the geometry matches all conditions"""
        for field, operator, value in conditions:
            if field not in geom:
                return False
            
            field_value = geom[field]
            if operator == '=':
                if not (field_value == value):
                    return False
            elif operator == '!=':
                if not (field_value != value):
                    return False
            elif operator == '>':
                if not (field_value > value):
                    return False
            elif operator == '>=':
                if not (field_value >= value):
                    return False
            elif operator == '<':
                if not (field_value < value):
                    return False
            elif operator == '<=':
                if not (field_value <= value):
                    return False
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        
        return True
    
    def delete(self, geom_id: int) -> bool:
        """
        Delete the geometry shape with the specified ID
        :param geom_id: ID of the shape to delete
        :return: Whether deletion was successful
        """
        for i, geom in enumerate(self.geometries):
            if geom['id'] == geom_id:
                del self.geometries[i]
                self.save_to_cache()
                return True
        return False
    
    def plot(self, geometries: List[Geometry], title: str = "GDB Geometry Plot"):
        """
        Plot geometry shapes, can handle partial field query results
        :param geometries: List of geometry shapes to plot
        :param title: Chart title
        """
        if not geometries:
            print("No geometries to plot")
            return
            
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True)
        
        for geom in geometries:
            if geom['type'] == 'point':
                if 'x' in geom and 'y' in geom:
                    ax.plot(geom['x'], geom['y'], 'ro')
                    ax.text(geom['x'], geom['y'], f"P{geom['id']}", fontsize=12)
            elif geom['type'] == 'line':
                if 'start' in geom and 'end' in geom:
                    start = geom['start']
                    end = geom['end']
                    ax.plot([start['x'], end['x']], [start['y'], end['y']], 'b-')
                    mid_x = (start['x'] + end['x']) / 2
                    mid_y = (start['y'] + end['y']) / 2
                    ax.text(mid_x, mid_y, f"L{geom['id']}", fontsize=12)
            elif geom['type'] == 'rectangle':
                if all(k in geom for k in ['x', 'y', 'width', 'height']):
                    rect = patches.Rectangle(
                        (geom['x'], geom['y']), geom['width'], geom['height'],
                        linewidth=1, edgecolor='g', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(geom['x'] + geom['width']/2, geom['y'] + geom['height']/2, 
                           f"R{geom['id']}", fontsize=12)
            elif geom['type'] == 'circle':
                if 'center' in geom and 'radius' in geom:
                    circle = patches.Circle(
                        (geom['center']['x'], geom['center']['y']), geom['radius'],
                        linewidth=1, edgecolor='purple', facecolor='none'
                    )
                    ax.add_patch(circle)
                    ax.text(geom['center']['x'], geom['center']['y'], 
                           f"C{geom['id']}", fontsize=12)
            elif geom['type'] == 'mesh_triangle':
                if 'vertices' in geom:
                    triangle = geom['vertices']
                    if len(triangle) == 3:
                        polygon = patches.Polygon(
                            [(v['x'], v['y']) for v in triangle], closed=True,
                            linewidth=1, edgecolor='orange', facecolor='none'
                        )
                        ax.add_patch(polygon)
                        ax.text(
                            sum(v['x'] for v in triangle) / 3,
                            sum(v['y'] for v in triangle) / 3,
                            f"T{geom['id']}", fontsize=12
                        )
        
        # Auto-adjust limits based on content
        x_vals = []
        y_vals = []
        for geom in geometries:
            if geom['type'] == 'point' and 'x' in geom and 'y' in geom:
                x_vals.append(geom['x'])
                y_vals.append(geom['y'])
            elif geom['type'] == 'line' and 'start' in geom and 'end' in geom:
                x_vals.extend([geom['start']['x'], geom['end']['x']])
                y_vals.extend([geom['start']['y'], geom['end']['y']])
            elif geom['type'] == 'rectangle' and all(k in geom for k in ['x', 'y', 'width', 'height']):
                x_vals.extend([geom['x'], geom['x'] + geom['width']])
                y_vals.extend([geom['y'], geom['y'] + geom['height']])
            elif geom['type'] == 'circle' and 'center' in geom and 'radius' in geom:
                x_vals.extend([geom['center']['x'] - geom['radius'], geom['center']['x'] + geom['radius']])
                y_vals.extend([geom['center']['y'] - geom['radius'], geom['center']['y'] + geom['radius']])
            elif geom['type'] == 'mesh_triangle' and 'vertices' in geom:
                x_vals.extend([v['x'] for v in geom['vertices']])
                y_vals.extend([v['y'] for v in geom['vertices']])
        
        if x_vals and y_vals:
            padding = 1.0  # Add some padding around the geometries
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a database instance with cache
    gdb = GDB('gdb_cache.pkl')
    
    # Add some geometry shapes
    gdb.add_geometry('point', x=1, y=1, color='red')
    gdb.add_geometry('point', x=3, y=4, color='blue')
    gdb.add_geometry('line', start={'x': 1, 'y': 1}, end={'x': 3, 'y': 4}, style='solid')
    gdb.add_geometry('rectangle', x=2, y=2, width=3, height=2, fill=False)
    gdb.add_geometry('circle', center={'x': 5, 'y': 5}, radius=2, fill=True)

    # Read mesh data from OBJ file (if it exists)
    obj_file = 'C:/Users/18858/Desktop/1.obj'
    if os.path.exists(obj_file):
        gdb.read_obj_file(obj_file)
    else:
        print(f"Warning: OBJ file not found at {obj_file}")

    print("=== GDB Spatial Geometry Database Demonstration ===")
    print(f"Database contains {len(gdb.geometries)} geometries\n")

    # Example SQL queries
    queries = [
        "SELECT * FROM geometries",
        "SELECT id, type, x, y FROM geometries WHERE type = 'point'",
        "SELECT id, type, vertices FROM geometries WHERE type = 'mesh_triangle'",
        "SELECT id, type FROM geometries WHERE type = 'circle' AND radius > 1"
    ]
    
    for query in queries:
        print(f"\nExecuting query: {query}")
        try:
            results = gdb.execute_sql(query)
            print(f"Found {len(results)} results:")
            for geom in results[:5]:  # Print first 5 results to avoid flooding
                print(geom)
            if len(results) > 5:
                print(f"... and {len(results)-5} more")
            
            # Plot query results
            gdb.plot(results, f"Query results: {query[:20]}...")
        except ValueError as e:
            print(f"Query error: {e}")