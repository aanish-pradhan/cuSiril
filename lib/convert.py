
# IMPORT PACKAGES
from pysiril.siril import Siril # Siril app
from pysiril.wrapper import Wrapper # Siril API
import sys

# LAUNCH SIRIL
app = Siril("/usr/bin/siril")
cmd = Wrapper(app)
app.Open()

# CONVERT DATA
dataPath = sys.argv[1]

cmd.cd(dataPath + "/Bias/")
cmd.convertraw(basename = "Bias", debayer = True, fitseq = True, 
	out = dataPath + "/Bias/")

cmd.cd(dataPath + "/Dark/")
cmd.convertraw(basename = "Dark", debayer = True, fitseq = True, 
	out = dataPath + "/Dark/")

cmd.cd(dataPath + "/Flat/")
cmd.convertraw(basename = "Flat", debayer = True, fitseq = True, 
	out = dataPath + "/Flat/")

cmd.cd(dataPath + "/Flat/")
cmd.convertraw(basename = "Light", debayer = True, fitseq = True, 
	out = dataPath + "/Light/")

# CLOSE SIRIL
app.Close()


