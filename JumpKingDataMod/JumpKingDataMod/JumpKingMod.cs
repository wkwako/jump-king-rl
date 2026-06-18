using EntityComponent;
using EntityComponent.BT;
using JumpKing;
using JumpKing.API;
using JumpKing.BodyCompBehaviours;
using JumpKing.Level;
using JumpKing.Mods;
using JumpKing.Player;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;

namespace JumpKingDataMod
{
    [JumpKingMod("wkwako.GameStateMod")]
    public static class GameStateMod
    {
        public static string _outputPath;
        private static GameStateWriterBehaviour _behaviour;

        [OnLevelStart]
        public static void OnLevelStart()
        {
            _outputPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "gamestate.txt"
            );

            PlayerEntity player = EntityManager.instance.Find<PlayerEntity>();
            if (player == null) return;
            _behaviour = new GameStateWriterBehaviour(player);
            player.m_body.RegisterBehaviour(_behaviour);

            // scan all screens once on load
            int totalScreens = LevelManager.TotalScreens;
            PlatformScanner.ScanAllScreens(totalScreens);
        }

        [OnLevelEnd]
        public static void OnLevelEnd()
        {
            if (_behaviour != null)
                _behaviour.Deactivate();
        }
    }

    public class GameStateWriterBehaviour : IBodyCompBehaviour
    {
        private bool _wasOnGround = false;
        private float? _maxHeightThisJump = null;
        private bool _isActive = true;
        private bool _reflectionInitialized = false;
        private int _prevScreen = -1;
        private int _writeCount = 0;
        private bool _firstAirborneFrame = false;
        private float _prevWindVelocity = 0f;
        private int _globalFrameCount = 0;
        private bool _cycleValid = false;
        private float _windTimer = 0f;
        private bool _prevWindNegative = false;
        const float DELTA_TIME = 1f / 60f;

        private Type _jumpChargeCalcType;
        private PropertyInfo _jumpFramesProp;
        private PropertyInfo _jumpPercentageProp;
        private TeleporterBehavior _teleporter;
        private FieldInfo _windQueryField;
        private FieldInfo _collisionQueryField;
        private MethodInfo _windGetVelocityMethod;
        private ActionKeylogger _keylogger;
        private WindCycleRecorder _windCycleRecorder;

        // TCP server fields
        private TcpListener _server;
        private TcpClient _client;
        private NetworkStream _stream;
        private Thread _listenThread;
        private const int PORT = 7777;

        // teleport command received from Python
        private float _teleportX = -1f;
        private float _teleportY = -1f;
        private readonly object _teleportLock = new object();

        private static readonly int[] WindScreens = { 25, 26, 27, 28, 29, 30, 31 };
        private static readonly int[] IceScreens = { 36, 37, 38 };

        public GameStateWriterBehaviour(PlayerEntity player)
        {
            _teleporter = new TeleporterBehavior();
            _keylogger = new ActionKeylogger();
            _windCycleRecorder = new WindCycleRecorder();
            StartServer();
        }

        private void StartServer()
        {
            try
            {
                _server = new TcpListener(IPAddress.Loopback, PORT);
                _server.Start();
                _listenThread = new Thread(ListenForClient);
                _listenThread.IsBackground = true;
                _listenThread.Start();
            }
            catch (Exception e)
            {
                File.AppendAllText(GameStateMod._outputPath + ".log", $"Server start error: {e.Message}\n");
            }
        }

        private void ListenForClient()
        {
            while (_isActive)
            {
                try
                {
                    // wait for Python to connect
                    _client = _server.AcceptTcpClient();
                    _client.NoDelay = true;
                    _stream = _client.GetStream();

                    // listen for teleport commands from Python
                    byte[] buffer = new byte[256];
                    while (_isActive && _client.Connected)
                    {
                        try
                        {
                            int bytesRead = _stream.Read(buffer, 0, buffer.Length);
                            if (bytesRead > 0)
                            {
                                string cmd = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                                // format: "teleport:x,y"
                                if (cmd.StartsWith("teleport:"))
                                {
                                    string[] parts = cmd.Substring(9).Split(',');
                                    if (parts.Length == 2 &&
                                        float.TryParse(parts[0], out float tx) &&
                                        float.TryParse(parts[1], out float ty))
                                    {
                                        lock (_teleportLock)
                                        {
                                            _teleportX = tx;
                                            _teleportY = ty;
                                        }
                                    }
                                }
                            }
                        }
                        catch { break; }
                    }
                }
                catch (Exception e)
                {
                    if (_isActive)
                        File.AppendAllText(GameStateMod._outputPath + ".log", $"Client error: {e.Message}\n");
                    Thread.Sleep(100);
                }
            }
        }

        private void SendState(string state)
        {
            if (_stream == null || _client == null || !_client.Connected) return;
            try
            {
                // prefix with length so Python knows where message ends
                byte[] data = Encoding.UTF8.GetBytes(state + "\n");
                _stream.Write(data, 0, data.Length);
                _stream.Flush();
            }
            catch
            {
                // client disconnected, ignore
            }
        }

        private void WriteStateSafe(string state)
        {
            _writeCount++;
            // still write to file as fallback for keylogger and debugging
            int attempts = 0;
            while (attempts < 5)
            {
                try
                {
                    File.WriteAllText(GameStateMod._outputPath, state);
                    return;
                }
                catch (IOException)
                {
                    attempts++;
                    Thread.Sleep(5);
                }
                catch (UnauthorizedAccessException)
                {
                    return;
                }
            }
        }

        private void InitializeReflection()
        {

            foreach (Assembly asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (asm.GetName().Name == "JumpKingLastJumpValue")
                {
                    _jumpChargeCalcType = asm.GetType("JumpKingLastJumpValue.Models.JumpChargeCalc");
                    if (_jumpChargeCalcType != null)
                    {
                        _jumpFramesProp = _jumpChargeCalcType.GetProperty("JumpFrames", BindingFlags.Public | BindingFlags.Static);
                        _jumpPercentageProp = _jumpChargeCalcType.GetProperty("JumpPercentage", BindingFlags.Public | BindingFlags.Static);
                    }
                    break;
                }

                _windQueryField = typeof(BodyComp).GetField("m_windVelocityQuery",
                    BindingFlags.NonPublic | BindingFlags.Instance);
                _windGetVelocityMethod = typeof(WindManager).GetMethod(
                    "JumpKing.API.IWindVelocityQuery.GetCurrentVelocity",
                    BindingFlags.NonPublic | BindingFlags.Instance);
            }
            _reflectionInitialized = true;
        }

        public void Deactivate()
        {
            _isActive = false;
            _teleporter?.Deactivate();
            try { _stream?.Close(); } catch { }
            try { _client?.Close(); } catch { }
            try { _server?.Stop(); } catch { }
            try
            {
                if (File.Exists(GameStateMod._outputPath))
                    File.Delete(GameStateMod._outputPath);
            }
            catch { }
        }

        public bool ExecuteBehaviour(BehaviourContext context)
        {
            if (!_isActive) return true;

            if (!_reflectionInitialized)
                InitializeReflection();

            try
            {
                BodyComp body = context.BodyComp;
                if (body == null) return true;
                _teleporter?.Update(body);

                // handle teleport commands from Python
                lock (_teleportLock)
                {
                    if (_teleportX >= 0 && body.IsOnGround)
                    {
                        body.Position.X = _teleportX;
                        body.Position.Y = _teleportY;
                        body.Velocity = Vector2.Zero;
                        Camera.UpdateCamera(body.GetHitbox().Center);
                        _teleportX = -1f;
                        _teleportY = -1f;
                    }
                }

                bool isOnGround = body.IsOnGround;
                int currentScreen = Camera.CurrentScreen;
                float x = body.Position.X;
                float y = body.Position.Y;
                float velX = body.Velocity.X;
                float velY = body.Velocity.Y;
                int totalScreens = LevelManager.TotalScreens;

                int jumpFrames = _jumpFramesProp != null
                    ? (int)_jumpFramesProp.GetValue(null)
                    : 0;
                float jumpPercentage = _jumpPercentageProp != null
                    ? (float)_jumpPercentageProp.GetValue(null)
                    : 0f;

                // reset on first airborne frame
                if (!isOnGround && _wasOnGround)
                    _maxHeightThisJump = null;

                // track max height while airborne
                if (!isOnGround)
                {
                    if (_maxHeightThisJump == null || y < _maxHeightThisJump)
                        _maxHeightThisJump = y;
                }

                float maxHeight = _maxHeightThisJump ?? y;

                // get block states
                bool isOnIce = body.IsOnBlock(typeof(IceBlock));
                bool isInSnow = body.IsOnBlock(typeof(SnowBlock));
                bool isInWater = body.IsOnBlock(typeof(WaterBlock));

                // get wind velocity via reflection
                float windVelocity = 0f;
                if (_windQueryField != null && _windGetVelocityMethod != null)
                {
                    object windQuery = _windQueryField.GetValue(body);
                    if (windQuery != null)
                        windVelocity = (float)_windGetVelocityMethod.Invoke(windQuery, null);
                }

                float windAcceleration = windVelocity - _prevWindVelocity;
                _prevWindVelocity = windVelocity;

                bool windNegative = windVelocity < 0f;

                // detect negative -> positive crossing
                if (_prevWindNegative && !windNegative)
                {
                    _cycleValid = true;
                    _windTimer = 0f;
                }

                if (_cycleValid)
                    _windTimer += DELTA_TIME;

                _prevWindNegative = windNegative;

                float windTimerOut = _cycleValid ? _windTimer : -1f;
                string windTimerStr = (_cycleValid ? _windTimer : -1f).ToString("F2");
                string state = $@"{{""x"":{x:F2},""y"":{-y:F2},""vel_x"":{velX:F2},""vel_y"":{-velY:F2},""is_on_ground"":{isOnGround.ToString().ToLower()},""current_screen"":{currentScreen},""total_screens"":{totalScreens},""jump_frames"":{jumpFrames},""jump_percentage"":{jumpPercentage:F4},""max_height"":{-maxHeight:F2},""is_on_ice"":{isOnIce.ToString().ToLower()},""is_in_snow"":{isInSnow.ToString().ToLower()},""is_in_water"":{isInWater.ToString().ToLower()},""wind_velocity"":{windVelocity:F4},""wind_acceleration"":{windAcceleration:F6},""write_count"":{_writeCount}, ""frame_count"":{_globalFrameCount}, ""wind_timer"":{windTimerStr}}}";

                // keylogger runs every frame
                _keylogger.Update(state, isOnGround);

                // wind cycle recorder
                _windCycleRecorder.Update(windVelocity);

                // send every frame over socket
                SendState(state);

                // two-frame airborne detection for write trigger
                if (!isOnGround && _wasOnGround)
                {
                    _firstAirborneFrame = true;
                }
                else if (!isOnGround && _firstAirborneFrame)
                {
                    _writeCount++;
                    WriteStateSafe(state);
                    _firstAirborneFrame = false;
                }

                if (isOnGround && !_wasOnGround)
                {
                    // landing
                    WriteStateSafe(state);
                    _firstAirborneFrame = false;
                }

                if (isOnGround)
                    _firstAirborneFrame = false;

                // write on screen transition
                if (currentScreen != _prevScreen && _prevScreen != -1)
                    WriteStateSafe(state);

                _prevScreen = currentScreen;
                _wasOnGround = isOnGround;
            }
            catch (Exception e)
            {
                WriteStateSafe($"{{\"error\": \"{e.Message}\"}}");
            }

            _globalFrameCount++;

            return true;
        }
    }
}