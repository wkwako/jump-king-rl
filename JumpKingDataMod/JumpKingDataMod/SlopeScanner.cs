using JumpKing.Level;
using Microsoft.Xna.Framework;
using System;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;

namespace JumpKingDataMod
{
    public static class SlopeScanner
    {
        private static FieldInfo _hitboxField;
        private static FieldInfo _screensField;
        private static MethodInfo _getSlopeTypeMethod;
        private static bool _initialized = false;
        public static string _outputPath;

        private static void Initialize()
        {
            _hitboxField = typeof(LevelScreen).GetField("m_hitboxes",
                BindingFlags.NonPublic | BindingFlags.Instance);

            _screensField = typeof(LevelManager).GetField("m_screens",
                BindingFlags.NonPublic | BindingFlags.Static);

            _getSlopeTypeMethod = typeof(SlopeBlock).GetMethod("GetSlopeType",
                BindingFlags.Public | BindingFlags.Instance);

            _outputPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "slopedata.txt"
            );

            _initialized = true;
        }

        public static void ScanAllScreens(int totalScreens)
        {
            try
            {
                if (!_initialized)
                    Initialize();

                if (_hitboxField == null) return;

                StringBuilder sb = new StringBuilder();
                sb.AppendLine("{");

                for (int i = 0; i < totalScreens; i++)
                {
                    sb.Append($"  \"{i}\": [");

                    LevelScreen[] screens = (LevelScreen[])_screensField.GetValue(null);
                    if (screens != null && i < screens.Length && screens[i] != null)
                    {
                        IBlock[] hitboxes = (IBlock[])_hitboxField.GetValue(screens[i]);
                        if (hitboxes != null)
                        {
                            bool first = true;
                            foreach (IBlock block in hitboxes)
                            {
                                if (block == null) continue;
                                if (!(block is SlopeBlock)) continue;

                                SlopeBlock slope = (SlopeBlock)block;
                                Rectangle rect = slope.GetRect();
                                string slopeType = _getSlopeTypeMethod != null
                                    ? _getSlopeTypeMethod.Invoke(slope, null).ToString()
                                    : "Unknown";

                                if (!first) sb.Append(",");
                                sb.Append($"[{rect.X},{rect.Y},{rect.Width},{rect.Height},\"{slopeType}\"]");
                                first = false;
                            }
                        }
                    }

                    if (i < totalScreens - 1)
                        sb.AppendLine("],");
                    else
                        sb.AppendLine("]");
                }

                sb.AppendLine("}");
                WriteSafe(sb.ToString());
            }
            catch (Exception e)
            {
                WriteSafe($"{{\"error\": \"{e.Message}\"}}");
            }
        }

        private static void WriteSafe(string data)
        {
            int attempts = 0;
            while (attempts < 5)
            {
                try
                {
                    File.WriteAllText(_outputPath, data);
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
    }
}