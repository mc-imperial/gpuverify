VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  # if ENV["PROCESSOR_ARCHITECTURE"] == "x86"
    config.vm.box = "hashicorp/precise32"
    config.vm.provision :shell, :path => "Scripts/Vagrant/bootstrap.sh", :args => "x86"
  # else
  #   config.vm.box = "hashicorp/precise64"
  #   config.vm.provision :shell, :path => "Scripts/Vagrant/bootstrap.sh", :args => "x86_64"
  # end
    
end
