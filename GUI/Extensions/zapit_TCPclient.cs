using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Net.Sockets;
using Bonsai.Reactive;
using System.Security.Cryptography;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class zapit_TCPclient
{
    public zapit_TCPclient()
    {
        tcp_port    = 1024;
        tcp_ip      = "127.0.0.1";
        buffer_size = 16;
        connected   = false;        
    }
    private Tuple<double,byte,byte,byte> connect(int tcp_port, string tcp_ip, int buffer_size)
    {
        this.client = new TcpClient(tcp_ip,tcp_port);
        client.ReceiveBufferSize = buffer_size;
        this.nwStream = client.GetStream();
        this.connected = true;
        Console.WriteLine("Connected to port " + this.tcp_port + " at address " + this.tcp_ip);
        return Tuple.Create(1.0,(byte)1, (byte)1, (byte)1);
    }
    private Tuple<double,byte,byte,byte> send_receive(Tuple<byte, byte, byte, byte, float, float, float> message)
    {
        var message_array = new byte[16];
        message_array[0] = message.Item1;
        message_array[1] = message.Item2;
        message_array[2] = message.Item3;
        message_array[3] = message.Item4;
        // Convert float values to byte arrays
        byte[] item5Bytes = BitConverter.GetBytes(message.Item5);
        Array.Copy(item5Bytes, 0, message_array, 4, item5Bytes.Length);

        byte[] item6Bytes = BitConverter.GetBytes(message.Item6);
        Array.Copy(item6Bytes, 0, message_array, 8, item6Bytes.Length);

        byte[] item7Bytes = BitConverter.GetBytes(message.Item7);
        Array.Copy(item7Bytes, 0, message_array, 12, item7Bytes.Length);
        Console.WriteLine(message_array.Length);
        this.nwStream.Write(message_array, 0, message_array.Length);
        // Receive the response
        byte[] bytesToRead = new byte[this.client.ReceiveBufferSize];
        // Process incoming message
        var bytesRead = this.nwStream.Read(bytesToRead, 0, bytesToRead.Length);
        var timestamp = BitConverter.ToDouble(bytesToRead,0);
        var comm_byte = bytesToRead[8];
        var resp_byte0 = bytesToRead[9];
        var resp_byte1 = bytesToRead[10];
        return Tuple.Create(timestamp, comm_byte, resp_byte0, resp_byte1);
    }
    /// <summary>
    /// Gets or sets the port for tcp communication
    /// </summary
    [Description("The tcp port for the server connection")]
    public int tcp_port {get; set;}

    /// <summary>
    /// Gets or sets the ip address for tcp communication
    /// </summary>
    [Description("The tcp-ip address for communication.  Default is LocalHost")]
    public string tcp_ip { get; set; }

    /// <summary>
    /// Gets or sets the maximum buffer size for receiving messages
    /// </summary>
    [Description("The maximum buffer size for receiving messages")]
    public int buffer_size { get; set; }

    private bool connected { get; set; }
    private TcpClient client { get; set; }
    private NetworkStream nwStream { get; set; }
    public IObservable<Tuple<double,byte,byte,byte>> Process(IObservable<Tuple<byte,byte,byte,byte, float, float, float>> source)
    {
        return source.Select(val =>
        {
            if (val.Item1 == 255 && !this.connected)
            {
                // Connect command with the client disconnected
                return connect(this.tcp_port, this.tcp_ip, this.buffer_size);
            }
            else if (val.Item1 < 254 && this.connected)
            {
                // Message to be sent whilst the client is connected
                return send_receive(val);
            }
            else if (val.Item1 == 255 && this.connected)
            {
                // Connect command but the client is already connected!
                return Tuple.Create(-1.0, (byte)0, (byte)1, (byte)0);
            }
            else if (val.Item1 == 254 && this.connected)
            {
                // Command to disconnect the client whilst it is connected
                this.client.Close();
                Console.WriteLine("Connection to port " + this.tcp_port + " at address " + this.tcp_ip + " closed");
                this.connected = false;
                return Tuple.Create(-2.0, (byte)1, (byte)0, (byte)0);
            }
            else if (val.Item1 == 254 && !this.connected)
            {
                // Command to disconnect the client whilst it is disconnected
                return Tuple.Create(-1.0, (byte)0, (byte)1, (byte)0);
            }

            // Unrecognised signal (i.e., message to be sent but client is not connected
            return Tuple.Create(-1.0, (byte)0, (byte)0, (byte)1);
        });
    }
    ~zapit_TCPclient()
    {
        if (this.connected)
        {
            this.client.Close();
            this.connected = false;
        }
    }
}