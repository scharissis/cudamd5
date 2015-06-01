# CUDA Installation Guide #

  * Download Cuda (drivers,toolkit,sdk)

  * Install drivers
    * Uninstall old drivers (I used 'envyng -t' to do this) ([important!])
    * chmod +x NVIDIA-Linux-x86-185.18.14-pkg1.run
    * Ctrl+Alt+F3 & log-in
    * sudo /etc/init.d/gdm stop
    * sudo ./NVIDIA-Linux-x86-185.18.14-pkg1.run (compile kernel and allow xconf)
    * sudo shutdown -r now

  * Install Toolkit
    * chmod +x cudatoolkit\_2.2\_linux\_32\_rhel5.3.run
    * sudo ./cudatoolkit\_2.2\_linux\_32\_rhel5.3.run

  * Install SDK
    * chmod +x cudasdk\_2.21\_linux.run
    * sudo ./cudasdk\_2.21\_linux.run

  * Export Paths
    * export PATH=$PATH:/usr/local/cuda/bin
    * export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:/usr/local/cuda/lib

  * Install Packages ([to compile SDK examples](Needed.md))
    * sudo apt-get install freeglut3-dev
    * sudo apt-get install build-essential
    * sudo apt-get install libxi-dev
    * sudo apt-get install libxmu-dev
    * sudo apt-get install libqt4-dev          (for cudaprof, at least as of CUDA 3.0)