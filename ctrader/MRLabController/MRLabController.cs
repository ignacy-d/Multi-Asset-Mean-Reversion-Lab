// Operational parameters for the generated native Python cBot wrapper.
// M5A deliberately has no trading-enable parameter or strategy tuning surface.
using cAlgo.API;

namespace cAlgo.Robots;

public partial class MRLabController : Robot
{
    [Parameter("Mode", DefaultValue = "OBSERVATION_ONLY")]
    public string Mode { get; set; }

    [Parameter("Symbols CSV", DefaultValue = "EURUSD,GBPUSD")]
    public string SymbolsCsv { get; set; }

    [Parameter("Heartbeat Seconds", DefaultValue = 10, MinValue = 1)]
    public int HeartbeatSeconds { get; set; }

    [Parameter("Log Verbosity", DefaultValue = "INFO")]
    public string LogVerbosity { get; set; }
}
