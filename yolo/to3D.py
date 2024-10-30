""" 
Subscriber to "aligned".
Also has a stack containining aligned left, right (boxes, confs, class_ids) pairs.
Pops the first tuple of pairs:
For each object, does the math, combines and converts the 2D boxes from left and right 
into one 3D box, and returns a single (3Dboxes, confs, class_ids) triple.
"""