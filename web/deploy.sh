#!/bin/sh

# Deploys the site locally
sudo gatsby build --prefix-paths
sudo cp -fr public/* /var/www/html/eyeofgollum/.
