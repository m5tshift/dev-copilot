# Keeps Windows awake while a long job runs, without changing power settings:
# SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) until the flag file exists.
# An eval chain once stopped after 11 of 116 runs because the PC went to sleep.
#
#   start "" /b powershell -File keepawake.ps1 C:\src\job.done   (then create job.done at the end)
param([Parameter(Mandatory)][string]$Flag)
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
while (-not (Test-Path $Flag)) { Start-Sleep 30 }
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
