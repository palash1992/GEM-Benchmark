########################
Installation
########################

gemben is available in the PyPi's repository and is pip installable. Please follow the following steps to install the library.

**Prepare your environment**::

    $ sudo apt update
    $ sudo apt install python3-dev python3-pip
    $ sudo pip3 install -U virtualenv

**Create a virtual environment**

If you have tensorflow installed in the root env, do the following::

    $ virtualenv --system-site-packages -p python3 ./venv

If you you want to install tensorflow later, do the following::

    $ virtualenv -p python3 ./venv

Activate the virtual environment using a shell-specific command::

    $ source ./venv/bin/activate

**Upgrade pip**::

    $ pip install --upgrade pip

**Install gemben using `pip`**::

    (venv) $ pip install gemben

**Install stable version directly from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/gemben.git
    (venv) $ cd gemben
    (venv) $ python setup.py install

**Install development version directly from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/gemben.git
    (venv) $ cd gemben
    (venv) $ git checkout development
    (venv) $ python setup.py install