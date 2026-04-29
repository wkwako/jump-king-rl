using JumpKing.Level;
using Microsoft.Xna.Framework;
using System;
using System.Reflection;
using System.Text;
using System.IO;
using System.Threading;
using JumpKing;

namespace JumpKingDataMod
{
    public static class PlatformScanner
    {
        private static FieldInfo _hitboxField;
        private static bool _initialized = false;
        public static string _outputPath;
        private static FieldInfo _screensField;

        private static void Initialize()
        {
            _hitboxField = typeof(LevelScreen).GetField("m_hitboxes",
                BindingFlags.NonPublic | BindingFlags.Instance);

            _screensField = typeof(LevelManager).GetField("m_screens",
                BindingFlags.NonPublic | BindingFlags.Static);

            _outputPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "platformdata.txt"
            );

            _initialized = true;
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

        public static void ScanAndWrite(float playerX, float playerY, int currentScreen, int totalScreens)
        {
            try
            {
                if (!_initialized)
                    Initialize();

                if (_hitboxField == null) return;

                StringBuilder sb = new StringBuilder();

                sb.AppendLine($"player:{playerX:F1},{playerY:F1}");
                sb.AppendLine($"screen:{currentScreen}");

                ScanScreen(playerX, playerY, currentScreen, 0, sb);

                if (currentScreen + 1 < totalScreens)
                {
                    sb.AppendLine($"screen:{currentScreen + 1}");
                    ScanScreen(playerX, playerY, currentScreen + 1, -360, sb);
                }

                WriteSafe(sb.ToString());
            }
            catch (Exception e)
            {
                WriteSafe($"ERR:{e.Message}");
            }
        }

        private static void ScanScreen(float playerX, float playerY, int screenIndex, float yOffset, StringBuilder sb)
        {
            if (_screensField == null) return;

            LevelScreen[] screens = (LevelScreen[])_screensField.GetValue(null);
            if (screens == null || screenIndex >= screens.Length) return;

            LevelScreen screen = screens[screenIndex];
            if (screen == null) return;

            IBlock[] hitboxes = (IBlock[])_hitboxField.GetValue(screen);
            if (hitboxes == null) return;

            foreach (IBlock block in hitboxes)
            {
                if (block == null) continue;
                Rectangle rect = block.GetRect();

                float relX = rect.X - playerX;
                float relY = (rect.Y - playerY) + yOffset;

                sb.AppendLine($"{relX:F0},{relY:F0},{rect.Width},{rect.Height}");
            }
        }

    }
}