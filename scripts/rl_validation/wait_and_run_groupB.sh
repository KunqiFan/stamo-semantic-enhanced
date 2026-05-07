#!/bin/bash
# Wait for Group A to finish (5000 step checkpoint appears), then launch Group B
cd "/c/Users/ryanf/Desktop/stamo_pro - 副本 - 副本"

echo "Waiting for Group A 5000-step checkpoint..."
while [ ! -d "StaMo/ckpts/maniskill_pickcube_v1_diffonly/5000" ]; do
    sleep 60
done
echo "Group A done! Starting Group B..."

py rl_validation/scripts/train_stamo_maniskill.py --group B --task PickCube-v1 --resume
