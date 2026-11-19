# PyMOL script: De Novo Protein Binder Complex
load results/top_lead_complex.pdb, complex
hide everything, complex
show cartoon, complex

# Color chains
color white, chain A               # Target (e.g. PD-L1)
color deepsalmon, chain B          # De Novo Designed Binder

# Highlight target hotspot epitope
select target_hotspots, chain A and resi 19+23+26+54+56+58+66+76+113+115+117+119+120+121+122+123+124+125
show sticks, target_hotspots
color marine, target_hotspots and name C*

# Highlight binder interface contacts (< 5.0 Å)
select binder_interface, chain B within 5.0 of chain A
show sticks, binder_interface
color firebrick, binder_interface and name C*

# Show interface polar contacts
distance interface_hbonds, target_hotspots, binder_interface, mode=2
set dash_color, yellow, interface_hbonds
set dash_width, 2.5, interface_hbonds

zoom chain B, 6.0
set ray_shadow, 0
bg_color white
