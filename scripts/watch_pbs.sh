#!/bin/bash

watch -n 10 "sshpass -p $(pass show imperial) ssh cx3 '/opt/pbs/bin/qstat -u \$USER'"