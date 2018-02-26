#!/bin/sh

CWD=`dirname $0`
WORKSPACE=`realpath $CWD`

docker run -it --privileged \
	--device=/dev/dri/card0:/dev/dri/card0:rwm \
	--device=/dev/dri/renderD128:/dev/dri/renderD128:rwm \
	--rm=true \
	-v $WORKSPACE:$WORKSPACE:rw \
	-v $HOME:$HOME:ro \
	-w $WORKSPACE \
	ubuntu-demo-opencl:beignet2 /bin/bash
