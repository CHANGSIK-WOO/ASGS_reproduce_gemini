from pyvis.network import Network
import os

FOC_SUPER_CLASS = {
    "air_small": ["plane", "airplane", "aircraft", "Fixed-wing Aircraft", "Small Aircraft", "Cargo Plane", "Helicopter", "helicopter", "A220", "A321", "A330", "A350", "ARJ21", "Boeing737", "Boeing747", "Boeing777", "Boeing787", "C919", "other airplane"],
    "ground_vehicle": ["vehicle", "large vehicle", "small vehicle", "Bus", "Cargo Truck", "Dump Truck", "Excavator", "Small Car", "Tractor", "Trailer", "Truck Tractor", "Van", "other vehicle", "Passenger Vehicle", "Pickup Truck", "Utility Truck", "Truck", "Truck w/Box", "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck", "Haul Truck", "Scraper/Tractor", "Front loader/Bulldozer", "Cement Mixer", "Ground Grader", "Railway Vehicle", "Passenger Car", "Locomotive", "Cargo Car", "Flat Car", "Tank car"],
    "vessel": ["ship", "Dry Cargo Ship", "Liquid Cargo Ship", "Passenger Ship", "Fishing Boat", "Motorboat", "Tugboat", "Warship", "Engineering Ship", "other ship", "Maritime Vessel", "Sailboat", "Barge", "Fishing Vessel", "Ferry", "Yacht", "Container Ship", "Oil Tanker"],
    "industrial_structure": ["storage tank", "storagetank", "Storage Tank", "oiltank", "working condensing tower", "unworking condensing tower", "working chimney", "unworking chimney", "chimney", "windmill", "Pylon", "Tower", "container crane", "Container Crane", "Tower crane", "Mobile Crane", "Reach Stacker", "Straddle Carrier", "Shipping Container", "Hut/Tent", "Shed"],
    "field_planar": ["baseball diamond", "tennis court", "basketball court", "ground track field", "soccer ball field", "swimming pool", "baseballfield", "tenniscourt", "basketballcourt", "groundtrackfield", "golffield", "stadium", "Baseball Field", "Basketball Court", "Tennis Court", "Football Field", "playground"],
    "site_infra": ["airport", "harbor", "helipad", "Helipad", "Aircraft Hangar", "bridge", "Bridge", "overpass", "roundabout", "Roundabout", "Intersection", "expressway service area", "expressway toll station", "trainstation", "dam", "Building", "Damaged Building", "Facility", "Construction Site", "Vehicle Lot", "Shipping container lot"],
}

# 1. 네트워크 초기화 (노트북에서 실행 시 notebook=True, 아니면 False)
# height와 width로 캔버스 크기 조절, bgcolor로 배경색 지정 가능
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)

# 물리 엔진 설정 (노드들이 서로 밀어내는 효과)
net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)

# 2. 노드와 엣지 추가
for super_class, sub_classes in FOC_SUPER_CLASS.items():
    # Super Class 노드 추가 (중심 노드, 색상과 크기를 다르게 설정)
    net.add_node(super_class, label=super_class, title=super_class, color='#FF5733', size=25, shape='star')

    for sub_class in sub_classes:
        # Sub Class 노드 추가 (말단 노드)
        # title 속성은 마우스를 올렸을 때 툴팁으로 표시됩니다.
        net.add_node(sub_class, label=sub_class, title=f"{sub_class} (belongs to {super_class})", color='#3498DB',
                     size=15)
        # 엣지 연결 (Super -> Sub)
        net.add_edge(super_class, sub_class, color='gray')

# 3. 결과 저장 및 표시
output_file = "foc_graph.html"
net.show(output_file, notebook=False)

# (선택사항) 코랩이나 주피터 노트북 환경에서 바로 열기 위한 코드
# import IPython
# IPython.display.HTML(filename=output_file)

print(f"그래프가 '{output_file}'로 저장되었습니다. 웹 브라우저로 열어보세요.")