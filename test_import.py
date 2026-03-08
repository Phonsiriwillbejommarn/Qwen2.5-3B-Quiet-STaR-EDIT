import faulthandler
faulthandler.enable()
print("Starting import...")
try:
    import modeling_quiet_star
    print("Success!")
except Exception as e:
    print("Error:", e)
